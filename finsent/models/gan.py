"""
WGAN-GP for Crisis Event Data Augmentation.
=============================================

Addresses H2: GAN-augmented training data on black-swan events improves 
model robustness as measured by Sharpe Ratio on out-of-sample crisis periods.

Problem:
    Financial crises are rare events (~5% of data) but cause the most damage.
    Models trained on mostly bull/normal market data are brittle during crashes.
    
Solution:
    Use Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate
    synthetic crisis-regime price sequences, conditioned on regime labels.

Architecture:
    Generator: noise_dim + regime_embedding → MLP → synthetic price window
    Critic: price window + regime_embedding → MLP → Wasserstein score
    
Why WGAN-GP over vanilla GAN:
    1. Wasserstein distance provides meaningful loss metric (correlates with quality)
    2. No mode collapse (critical — we need diverse crisis scenarios, not copies)
    3. Gradient penalty enforces Lipschitz constraint more stably than clipping
    4. Stable training without careful architecture balancing

Mathematical Framework:
    WGAN objective:
        min_G max_D  E[D(x)] - E[D(G(z))] + λ * E[(‖∇D(x̂)‖₂ - 1)²]
    
    where x̂ = εx + (1-ε)G(z), ε ~ U(0,1)  (gradient penalty)
    
    Regime conditioning:
        G(z, r) and D(x, r) where r ∈ {bull, bear, crisis, high_vol, low_vol}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List


class RegimeEmbedding(nn.Module):
    """Learnable embedding for market regime conditioning."""
    
    def __init__(self, n_regimes: int = 5, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(n_regimes, embedding_dim)
    
    def forward(self, regime_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(regime_ids)


class Generator(nn.Module):
    """WGAN-GP Generator — produces synthetic price windows.
    
    Architecture:
        [noise ⊕ regime_embedding] → Dense → BN → LeakyReLU → ... → Tanh → price window
    
    Output is reshaped to (batch, window_size, n_features) to match
    the price branch input format.
    """
    
    def __init__(
        self,
        noise_dim: int = 128,
        regime_embedding_dim: int = 32,
        hidden_dims: List[int] = None,
        output_dim: int = 450,  # window_size * n_features
        window_size: int = 30,
        n_features: int = 15,
        n_regimes: int = 5,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        self.window_size = window_size
        self.n_features = n_features
        self.output_dim = window_size * n_features
        
        self.regime_embed = RegimeEmbedding(n_regimes, regime_embedding_dim)
        
        input_dim = noise_dim + regime_embedding_dim
        
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, self.output_dim))
        layers.append(nn.Tanh())  # normalized to [-1, 1]
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        noise: torch.Tensor,       # (batch, noise_dim)
        regime: torch.Tensor,      # (batch,) integer regime IDs
    ) -> torch.Tensor:
        """
        Returns:
            synthetic: (batch, window_size, n_features, ) — synthetic price window
        """
        regime_emb = self.regime_embed(regime)  # (batch, regime_embed_dim)
        x = torch.cat([noise, regime_emb], dim=-1)
        flat = self.net(x)  # (batch, output_dim)
        return flat.view(-1, self.window_size, self.n_features)


class Critic(nn.Module):
    """WGAN-GP Critic (Discriminator) — scores real vs fake price windows.
    
    Note: In WGAN, the discriminator is called "critic" because it outputs
    an unbounded real-valued score, not a probability.
    
    Architecture:
        [flattened_price ⊕ regime_embedding] → Dense → LayerNorm → LeakyReLU → ... → score
    
    NO sigmoid output — Wasserstein distance is unbounded.
    Uses LayerNorm instead of BatchNorm (more stable for WGAN critics).
    """
    
    def __init__(
        self,
        input_dim: int = 450,  # window_size * n_features
        regime_embedding_dim: int = 32,
        hidden_dims: List[int] = None,
        n_regimes: int = 5,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        self.regime_embed = RegimeEmbedding(n_regimes, regime_embedding_dim)
        
        dim = input_dim + regime_embedding_dim
        
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(dim, h_dim),
                nn.LayerNorm(h_dim),  # LayerNorm, not BatchNorm
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
            dim = h_dim
        
        layers.append(nn.Linear(dim, 1))  # No activation — raw score
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        price_window: torch.Tensor,  # (batch, window_size, n_features)
        regime: torch.Tensor,         # (batch,) integer regime IDs
    ) -> torch.Tensor:
        """
        Returns:
            score: (batch, 1) — critic score (higher = more real)
        """
        flat = price_window.view(price_window.size(0), -1)
        regime_emb = self.regime_embed(regime)
        x = torch.cat([flat, regime_emb], dim=-1)
        return self.net(x)


class WGAN_GP:
    """WGAN-GP training and generation manager.
    
    Implements the full WGAN-GP training loop with:
    - Gradient penalty for Lipschitz constraint
    - Critic training with 5 steps per generator step
    - Regime-conditioned generation
    - Quality metrics (FID-like for time series)
    """
    
    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        noise_dim: int = 128,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lr_g: float = 1e-4,
        lr_c: float = 4e-4,
        beta1: float = 0.0,
        beta2: float = 0.9,
        device: str = "cuda",
    ):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.device = device
        
        # Separate optimizers (common WGAN practice)
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.opt_c = torch.optim.Adam(
            self.critic.parameters(), lr=lr_c, betas=(beta1, beta2)
        )
    
    def compute_gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient penalty: λ * E[(‖∇_x̂ D(x̂)‖₂ - 1)²]
        
        x̂ = ε·real + (1-ε)·fake  (random interpolation)
        
        Enforces the Lipschitz constraint that the critic has
        gradient norm ≤ 1 everywhere in the data manifold.
        """
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, device=self.device)
        
        # Interpolated samples
        interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
        
        # Critic scores on interpolated
        scores = self.critic(interpolated, regime)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Flatten and compute norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty: (norm - 1)²
        penalty = self.lambda_gp * ((gradient_norm - 1.0) ** 2).mean()
        
        return penalty
    
    def train_step(
        self,
        real_data: torch.Tensor,    # (batch, window_size, n_features)
        regime_ids: torch.Tensor,   # (batch,) regime labels
    ) -> Dict[str, float]:
        """Single training step: N critic updates + 1 generator update.
        
        Returns:
            dict of loss metrics for logging
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        regime_ids = regime_ids.to(self.device)
        
        # ─── Train Critic (n_critic steps) ────────────────────────
        critic_losses = []
        for _ in range(self.n_critic):
            self.opt_c.zero_grad()
            
            # Generate fake data
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_data = self.generator(noise, regime_ids).detach()
            
            # Critic scores
            real_scores = self.critic(real_data, regime_ids)
            fake_scores = self.critic(fake_data, regime_ids)
            
            # Wasserstein loss: maximize E[D(real)] - E[D(fake)]
            # Equivalent to minimizing E[D(fake)] - E[D(real)]
            w_loss = fake_scores.mean() - real_scores.mean()
            
            # Gradient penalty
            gp = self.compute_gradient_penalty(real_data, fake_data, regime_ids)
            
            critic_loss = w_loss + gp
            critic_loss.backward()
            self.opt_c.step()
            
            critic_losses.append(w_loss.item())
        
        # ─── Train Generator ──────────────────────────────────────
        self.opt_g.zero_grad()
        
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_data = self.generator(noise, regime_ids)
        fake_scores = self.critic(fake_data, regime_ids)
        
        # Generator wants to maximize critic score on fakes
        g_loss = -fake_scores.mean()
        g_loss.backward()
        self.opt_g.step()
        
        return {
            "critic_loss": np.mean(critic_losses),
            "generator_loss": g_loss.item(),
            "wasserstein_distance": -np.mean(critic_losses),  # approximation
        }
    
    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        regime_id: int,
    ) -> torch.Tensor:
        """Generate synthetic price windows for a specific regime.
        
        Args:
            n_samples: Number of synthetic windows to generate.
            regime_id: Market regime to condition on (0-4).
        
        Returns:
            synthetic: (n_samples, window_size, n_features)
        """
        self.generator.eval()
        
        noise = torch.randn(n_samples, self.noise_dim, device=self.device)
        regimes = torch.full((n_samples,), regime_id, dtype=torch.long, device=self.device)
        
        synthetic = self.generator(noise, regimes)
        
        self.generator.train()
        return synthetic.cpu()
    
    def save(self, path: str) -> None:
        """Save generator and critic state."""
        torch.save({
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_c": self.opt_c.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load saved state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opt_g.load_state_dict(checkpoint["opt_g"])
        self.opt_c.load_state_dict(checkpoint["opt_c"])


class RegimeDetector:
    """Simple regime detection using volatility and returns.
    
    Classifies each period into one of 5 regimes:
        0: Bull   — positive returns, low-moderate vol
        1: Bear   — negative returns, moderate vol
        2: Crisis — negative returns, very high vol
        3: High Vol — any direction, high vol
        4: Low Vol  — any direction, very low vol
    
    Uses rolling statistics with lookback window (no look-ahead).
    """
    
    def __init__(
        self,
        vol_window: int = 21,
        vol_crisis_threshold: float = 2.0,  # std above mean
        vol_high_threshold: float = 1.0,
        vol_low_threshold: float = -1.0,
        ret_window: int = 21,
    ):
        self.vol_window = vol_window
        self.vol_crisis_threshold = vol_crisis_threshold
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold
        self.ret_window = ret_window
    
    def detect(self, close_prices: np.ndarray) -> np.ndarray:
        """Detect market regime at each timestep.
        
        Returns: array of regime IDs (0-4), length = len(close_prices)
        """
        n = len(close_prices)
        regimes = np.full(n, 4, dtype=np.int64)  # default: low vol
        
        # Compute rolling returns and volatility
        returns = np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))
        
        for t in range(self.vol_window, n):
            window_ret = returns[t - self.ret_window:t]
            window_vol = returns[t - self.vol_window:t]
            
            cum_ret = np.sum(window_ret)
            vol = np.std(window_vol) * np.sqrt(252)  # annualized
            
            # Compute z-score of current vol vs expanding history
            hist_vols = []
            for s in range(self.vol_window, t + 1):
                hist_vols.append(np.std(returns[s - self.vol_window:s]) * np.sqrt(252))
            
            hist_vols = np.array(hist_vols)
            vol_zscore = (vol - np.mean(hist_vols)) / (np.std(hist_vols) + 1e-8)
            
            # Classify
            if vol_zscore >= self.vol_crisis_threshold and cum_ret < 0:
                regimes[t] = 2  # Crisis
            elif vol_zscore >= self.vol_high_threshold:
                regimes[t] = 3  # High Vol
            elif vol_zscore <= self.vol_low_threshold:
                regimes[t] = 4  # Low Vol
            elif cum_ret > 0:
                regimes[t] = 0  # Bull
            else:
                regimes[t] = 1  # Bear
        
        return regimes
