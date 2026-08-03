"""Market regimes from a Gaussian hidden Markov model (Hamilton, 1989).

Replaces the V2 ``if/else`` ladder over moving averages and volatility thresholds. Three
reasons the change matters for the paper:

1. The thresholds in a hand-tuned ladder are themselves fitted parameters, chosen by
   looking at the data, and they are never counted in the trial budget.
2. A referee can cite Hamilton; nobody can cite "SMA50 > SMA200 and vol < 20%".
3. Conditioning the augmentation GAN on a fitted latent state is defensible; conditioning
   it on an arbitrary tag is not.

Implemented directly with Baum-Welch (forward-backward + EM) on a univariate or
multivariate Gaussian emission, in log space for numerical stability. Fitting is causal
by construction when ``predict_states_online`` is used: the filtered state at ``t`` uses
observations up to ``t`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GaussianHMM", "RegimeLabels", "fit_regimes"]

_EPS = 1e-300
_LOG_EPS = -700.0


def _logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True) + _EPS) + amax
    return out if keepdims else np.squeeze(out, axis=axis)


@dataclass
class GaussianHMM:
    """Gaussian-emission HMM fitted by Baum-Welch.

    Parameters
    ----------
    n_states
        Number of latent regimes. Three is the usual choice for equities: a calm bull,
        a choppy middle, and a high-volatility crisis state.
    """

    n_states: int = 3
    max_iter: int = 200
    tol: float = 1e-6
    seed: int = 0

    start_log_prob: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    log_transmat: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    means: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    covars: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    log_likelihood_: float = np.nan
    n_iter_: int = 0

    # -- internals -------------------------------------------------------------------
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """(n_obs, n_states) log N(x_t ; mu_k, Sigma_k)."""
        n, d = X.shape
        out = np.empty((n, self.n_states))
        for k in range(self.n_states):
            cov = self.covars[k] + 1e-8 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
            inv = np.linalg.inv(cov)
            diff = X - self.means[k]
            quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        return np.maximum(out, _LOG_EPS)

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        n = log_b.shape[0]
        log_alpha = np.empty((n, self.n_states))
        log_alpha[0] = self.start_log_prob + log_b[0]
        for t in range(1, n):
            log_alpha[t] = log_b[t] + _logsumexp(
                log_alpha[t - 1][:, None] + self.log_transmat, axis=0
            )
        return log_alpha, float(_logsumexp(log_alpha[-1]))

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        n = log_b.shape[0]
        log_beta = np.zeros((n, self.n_states))
        for t in range(n - 2, -1, -1):
            log_beta[t] = _logsumexp(
                self.log_transmat + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )
        return log_beta

    # -- API -------------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] < X.shape[1]:
            X = X.T
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        # Initialise by volatility-ordered quantiles so state 0 is the calmest regime;
        # this makes the fitted states comparable across walk-forward folds.
        order = np.argsort(np.abs(X[:, 0]))
        chunks = np.array_split(order, self.n_states)
        self.means = np.stack([X[c].mean(axis=0) for c in chunks])
        self.covars = np.stack(
            [np.cov(X[c].T).reshape(d, d) + 1e-6 * np.eye(d) for c in chunks]
        )
        self.start_log_prob = np.log(np.full(self.n_states, 1.0 / self.n_states))
        trans = np.full((self.n_states, self.n_states), 0.1 / max(self.n_states - 1, 1))
        np.fill_diagonal(trans, 0.9)
        self.log_transmat = np.log(trans)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            log_b = self._log_emission(X)
            log_alpha, ll = self._forward(log_b)
            log_beta = self._backward(log_b)

            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.full((self.n_states, self.n_states), -np.inf)
            for t in range(n - 1):
                m = (
                    log_alpha[t][:, None]
                    + self.log_transmat
                    + log_b[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                )
                m -= _logsumexp(m.ravel())
                log_xi = np.logaddexp(log_xi, m)

            self.start_log_prob = np.log(np.maximum(gamma[0], _EPS))
            trans = np.exp(log_xi)
            trans /= np.maximum(trans.sum(axis=1, keepdims=True), _EPS)
            self.log_transmat = np.log(np.maximum(trans, _EPS))

            weights = np.maximum(gamma.sum(axis=0), _EPS)
            self.means = (gamma.T @ X) / weights[:, None]
            for k in range(self.n_states):
                diff = X - self.means[k]
                self.covars[k] = (gamma[:, k][:, None] * diff).T @ diff / weights[k]
                self.covars[k] += 1e-8 * np.eye(d)

            self.n_iter_ = it + 1
            self.log_likelihood_ = ll
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._reorder_by_volatility()
        return self

    def _reorder_by_volatility(self) -> None:
        """Sort states by emission volatility so labels are stable across folds."""
        vol = np.array([np.sqrt(np.trace(c)) for c in self.covars])
        order = np.argsort(vol)
        self.means = self.means[order]
        self.covars = self.covars[order]
        self.start_log_prob = self.start_log_prob[order]
        self.log_transmat = self.log_transmat[np.ix_(order, order)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed state posteriors. Uses the whole series -- not causal."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] < X.shape[1]:
            X = X.T
        log_b = self._log_emission(X)
        log_alpha, _ = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def predict_states_online(self, X: np.ndarray) -> np.ndarray:
        """**Causal** filtered state at each ``t``, using observations up to ``t`` only.

        This is the version any feature or conditioning variable must use. The smoothed
        posterior from ``predict_proba`` peeks at the future and would leak.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] < X.shape[1]:
            X = X.T
        log_b = self._log_emission(X)
        log_alpha, _ = self._forward(log_b)
        log_alpha = log_alpha - _logsumexp(log_alpha, axis=1, keepdims=True)
        return np.argmax(log_alpha, axis=1)


@dataclass(frozen=True)
class RegimeLabels:
    states: np.ndarray
    n_states: int
    state_vol: np.ndarray
    state_mean: np.ndarray
    log_likelihood: float

    def name(self, state: int) -> str:
        """Human-readable tag, assigned from fitted moments rather than by hand."""
        if state == 0:
            return "calm"
        if state == self.n_states - 1:
            return "crisis"
        return "transitional"


def fit_regimes(
    returns: np.ndarray,
    n_states: int = 3,
    max_iter: int = 200,
    seed: int = 0,
    causal: bool = True,
) -> tuple[GaussianHMM, RegimeLabels]:
    """Fit a regime model on a return series and return causal state assignments."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    X = np.column_stack([r, np.abs(r)])

    model = GaussianHMM(n_states=n_states, max_iter=max_iter, seed=seed).fit(X)
    states = model.predict_states_online(X) if causal else model.predict_proba(X).argmax(axis=1)

    return model, RegimeLabels(
        states=states,
        n_states=n_states,
        state_vol=np.array([float(np.sqrt(c[0, 0])) for c in model.covars]),
        state_mean=np.array([float(m[0]) for m in model.means]),
        log_likelihood=model.log_likelihood_,
    )
