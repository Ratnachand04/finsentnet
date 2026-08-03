# FinSentNet — Complete Working Procedure & Mathematical Foundations

> Research-paper companion document. Every equation below is traced to the exact
> source file/function that implements it, then categorized as **(E)** existing &
> directly used, **(T)** tuned (modified from a standard form), or **(C)** custom-made
> for this project. Full derivations are given for each.

---

## PART I — THE EXACT MODEL IDEA (system narrative)

FinSentNet is a **cross-modal deep-learning trading-intelligence engine**. The user
journey and the model's internal response map one-to-one:

1. **User inputs** — investment amount `C`, currency, investment duration (horizon `H`),
   and risk appetite `ρ ∈ [0,1]`.
2. **Market & stock selection** — user picks markets (NYSE/NASDAQ/S&P500/NSE/BSE/Crypto…)
   and individual tickers. The system shows default fundamental + technical info and a
   **candlestick chart with overlaid indicator lines** (SMA-50/200, EMA-20/50, Bollinger
   Bands, plus RSI/MACD/Stochastic sub-panels).
3. **Live analysis layer** — for each selected stock the engine pulls OHLCV, computes ~35
   technical features, runs the trained per-ticker neural network, and draws **BUY / SELL /
   HOLD** markers with **entry** and **exit (target/stop)** lines on the candles.
4. **News & sentiment layer** — a scraping + FinBERT/lexicon pipeline pulls and scores news,
   producing a sentiment vector that the network fuses with price action.
5. **"Analyze" button → final plan** — the system returns a complete capital plan: how much
   to invest per stock, holding (cash) amount, number of shares per company, entry price,
   exit (target) price, stop-loss, total deployed capital, per-stock BUY/SELL/HOLD
   probabilities and confidence, Kelly fraction, portfolio weights, expected return,
   volatility, Sharpe, VaR/CVaR, and drawdown.

> **Disclaimer encoded in the design:** the network emits *model assumptions* (probabilities
> and magnitude estimates), never guaranteed values. All "prices" are model-projected targets.

**Architecture (two branches → fusion → dual head):**

```
News text ─► Text Branch  (FinBERT → TextCNN → 3× BiLSTM → Multi-Head Self-Attn) ─► s ∈ ℝ^512
Price+TA ─► Price Branch (Multi-scale Conv1D → Dilated TCN → 2× LSTM)           ─► P ∈ ℝ^{30×512}
                              │
                              ▼
        Cross-Modal Gated Attention Fusion  (Q=s, K=V=P, σ-gate)  ─► f ∈ ℝ^512
                              │
              ┌───────────────┴───────────────┐
        Direction head (3-class softmax)   Return head (scalar regressor)
        P(Up), P(Neutral), P(Down)         predicted forward return r̂
```

Implemented in `finsentnet_pro/backend/models/finsentnet_core.py` (production) and
`finsent/models/finsent_net.py` (research). GAN augmentation lives in `finsent/models/gan.py`.

---

## PART II — END-TO-END WORKING PROCEDURE (stage by stage)

### Stage 0 — Data ingestion
`market_data_fetcher.py` pulls OHLCV (Yahoo Finance / CSV). `news_sentiment_engine.py`
pulls headlines. SQL snapshots for ~hundreds of tickers live in `backend/data/sql_training_companies/`.

### Stage 1 — Feature engineering (`technical_indicators.py`)
From raw `[Open, High, Low, Close, Volume]` compute ~35 strictly-causal features: RSI(14,21),
MACD(12,26,9), Bollinger(20,2σ) + width + position, ATR(14), OBV, VWAP, Volume ratio,
EMA/SMA {5,10,20,50,200}, price-vs-MA, Golden Cross flag, Doji/Hammer/Engulfing patterns,
log-return, 20-day annualized volatility, Stochastic %K/%D. → **Eqs E1–E18**.

### Stage 2 — Windowing, normalization, labels (`training/dataset.py`)
- Sliding window of `W=30` trading days, `F=20` features → tensor `X ∈ ℝ^{30×20}`.
- **Causal z-score** normalization (per window) → **Eq E19**.
- **Direction label** via forward-return threshold (triple-barrier-lite) → **Eq C1**.
- Forward return magnitude becomes the regression target.

### Stage 3 — Text encoding (`models/text_branch.py`)
FinBERT WordPiece tokens → 768-d contextual embeddings (last 2 layers fine-tuned) →
TextCNN (kernels {2,3,4,5}) → 3-layer BiLSTM (256/dir) with variational dropout →
sinusoidal positional encoding → 8-head self-attention → CLS+mean-pool sentence vector
`s ∈ ℝ^512`. → **Eqs E20–E31**.

### Stage 4 — Price encoding (`models/price_branch.py`)
Multi-scale Conv1D (kernels {3,7,15}) → Dilated TCN (dilations {1,2,4,8}) + residual →
2-layer LSTM → GELU projection to 512-d. Outputs context `p` and full sequence
`P ∈ ℝ^{30×512}`. → **Eqs E32–E35**.

### Stage 5 — Cross-modal fusion (`models/cross_modal_fusion.py`) — **core innovation**
Sentiment vector **queries** price history; a learned sigmoid gate blends modalities.
→ **Eqs C2–C3**.

### Stage 6 — Dual-head output (`models/dual_head_output.py`)
Direction logits → softmax → `(P_up, P_neu, P_down)`; return head → scalar `r̂`. → **Eq E36**.

### Stage 7 — Training (`training/trainer.py`, `training/losses.py`, calibration, schedulers)
Production objective = CrossEntropy(direction) + ½·Huber(return). Research objective =
Focal + Confidence-calibration. AdamW + cosine-warmup, gradient clipping, early stop on
validation Sharpe, post-hoc temperature scaling. → **Eqs E37–E45, T1–T3, C4–C5**.

### Stage 8 — Signal generation (`models/signal_generator.py`)
Map `P_up` → {STRONG BUY…STRONG SELL}; ATR-based target/stop modulated by risk appetite;
risk-reward; Kelly position sizing; share quantity; technical score; regime tag.
→ **Eqs T4–T8, C6–C8**.

### Stage 9 — Portfolio & risk (`portfolio/*.py`, `api/routes/analysis.py`)
Mean-variance / max-Sharpe / min-variance / risk-parity weights; allocation table with fees;
VaR/CVaR/Sharpe/Sortino/Calmar/MaxDD risk report. → **Eqs E46–E58, T9**.

### Stage 10 — Final summary (the "Analyze" payload)
`api/routes/analysis.py::analyze_stocks` assembles per-stock signals, portfolio optimization,
allocation, backtest, and risk report into the JSON the dashboard renders.

---

# PART III — ALL MATHEMATICAL EQUATIONS, DERIVED & CATEGORIZED

Legend: **(E)** existing/standard · **(T)** tuned · **(C)** custom.

---

## SECTION A — EXISTING, DIRECTLY-USED EQUATIONS

### A.1 Technical-indicator mathematics

#### **E1 — Exponential Moving Average (EMA)** · `technical_indicators.py: ewm(span=…)`
$$\text{EMA}_t = \alpha\,x_t + (1-\alpha)\,\text{EMA}_{t-1}, \qquad \alpha = \frac{2}{N+1}$$
**Derivation.** Define the weighted average that geometrically discounts the past:
$\text{EMA}_t = \alpha\sum_{k=0}^{\infty}(1-\alpha)^k x_{t-k}$. Split off the $k=0$ term:
$\text{EMA}_t = \alpha x_t + \alpha\sum_{k=1}^{\infty}(1-\alpha)^k x_{t-k}
= \alpha x_t + (1-\alpha)\big[\alpha\sum_{j=0}^{\infty}(1-\alpha)^j x_{t-1-j}\big]
= \alpha x_t + (1-\alpha)\text{EMA}_{t-1}$.
The "span" convention sets $\alpha = 2/(N+1)$ so the EMA's center of mass equals that of an
$N$-period SMA. Center of mass $= \sum_k k\,\alpha(1-\alpha)^k = (1-\alpha)/\alpha = (N-1)/2$. ∎

#### **E2 — Wilder's smoothing (RMA)** · `rsi()`, `atr()` use `ewm(alpha=1/period)`
$$\text{RMA}_t = \tfrac{1}{N}x_t + \big(1-\tfrac{1}{N}\big)\text{RMA}_{t-1}$$
Special case of E1 with $\alpha = 1/N$. Wilder's original recursive form
$\text{RMA}_t = \text{RMA}_{t-1} + \frac1N (x_t - \text{RMA}_{t-1})$ is algebraically identical.

#### **E3 — Relative Strength Index (RSI)** · `rsi()`
$$U_t=\max(\Delta_t,0),\quad D_t=\max(-\Delta_t,0),\quad \Delta_t = C_t-C_{t-1}$$
$$\overline{U}_t=\text{RMA}_N(U),\quad \overline{D}_t=\text{RMA}_N(D),\quad
RS_t=\frac{\overline{U}_t}{\overline{D}_t},\quad \boxed{RSI_t = 100 - \frac{100}{1+RS_t}}$$
**Derivation.** We want a bounded oscillator on $[0,100]$ expressing the share of upward
momentum. Let the "relative strength" be the ratio of average gains to average losses $RS$.
The up-fraction is $\frac{\overline U}{\overline U+\overline D}=\frac{RS}{1+RS}$. Scaling to
100 and rearranging: $100\cdot\frac{RS}{1+RS}=100\big(1-\frac{1}{1+RS}\big)$. ∎ (Wilder, 1978.)

#### **E4 — MACD, Signal, Histogram** · `macd()`
$$\text{MACD}_t = \text{EMA}_{12}(C)_t - \text{EMA}_{26}(C)_t,\quad
\text{Signal}_t = \text{EMA}_9(\text{MACD})_t,\quad
\text{Hist}_t = \text{MACD}_t - \text{Signal}_t$$
Difference of two EMAs (E1). The fast EMA reacts quicker; the sign of the histogram detects
momentum cross-overs (Appel, 1979).

#### **E5 — Simple Moving Average (SMA)** · `rolling(N).mean()`
$$\text{SMA}_N(x)_t = \frac1N\sum_{k=0}^{N-1}x_{t-k}$$

#### **E6 — Bollinger Bands** · `bollinger_bands()`
$$\text{Mid}_t = \text{SMA}_{20}(C)_t,\quad
\sigma_t=\sqrt{\tfrac1N\sum_{k=0}^{N-1}(C_{t-k}-\text{Mid}_t)^2},\quad
\text{Up/Lo}_t = \text{Mid}_t \pm 2\sigma_t$$
Bands are a $\pm2$-standard-deviation envelope; under approximate normality ~95% of prices fall
inside (Bollinger, 1980s).

#### **E7 — Average True Range (ATR)** · `atr()`
$$TR_t = \max\!\big(H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|\big),\qquad
\text{ATR}_t = \text{RMA}_{14}(TR)_t$$
True range is the largest of the three gaps that can occur across a session boundary;
Wilder-smoothing (E2) yields ATR. Used downstream for stop/target distances.

#### **E8 — On-Balance Volume (OBV)** · `obv()`
$$\text{OBV}_t = \text{OBV}_{t-1} + \operatorname{sgn}(C_t-C_{t-1})\,V_t$$
Cumulative signed volume; a running sum of volume flow (Granville, 1963).

#### **E9 — VWAP** · `vwap()`
$$\text{TP}_t=\tfrac{H_t+L_t+C_t}{3},\qquad
\text{VWAP}_t=\frac{\sum_{k\le t}\text{TP}_k V_k}{\sum_{k\le t}V_k}$$
Volume-weighted mean of the typical price — the average fill price if you traded proportional
to volume.

#### **E10 — Stochastic Oscillator (%K, %D)** · `stochastic()`
$$\%K_t = 100\cdot\frac{C_t-\min_{N}L}{\max_{N}H-\min_{N}L},\qquad
\%D_t=\text{SMA}_3(\%K)_t$$
Position of close within its recent high-low range (Lane, 1950s). $\%D$ smooths $\%K$.

#### **E11 — Log return & E12 — Annualized realized volatility** · `compute_all()`
$$r_t=\ln\frac{C_t}{C_{t-1}},\qquad
\sigma^{\text{ann}}_t = \operatorname{std}_{20}(r)\cdot\sqrt{252}$$
**Derivation of $\sqrt{252}$.** For i.i.d. daily returns, variance is additive over $n$ days:
$\operatorname{Var}(\sum_{i=1}^n r_i)=n\sigma_d^2$, so $\sigma_{n} = \sigma_d\sqrt n$. With 252
trading days/year, $\sigma^{\text{ann}}=\sigma_d\sqrt{252}$. ∎

### A.2 Neural-network mathematics

#### **E20 — Sinusoidal Positional Encoding** · `text_branch.py: PositionalEncoding`
$$PE_{(pos,2i)}=\sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\quad
PE_{(pos,2i+1)}=\cos\!\Big(\frac{pos}{10000^{2i/d}}\Big)$$
Each dimension is a sinusoid of geometrically increasing wavelength
($2\pi \to 2\pi\cdot10000$), letting attention express relative offsets: $PE_{pos+k}$ is a
fixed linear function of $PE_{pos}$ (Vaswani et al., 2017).

#### **E21 — Scaled Dot-Product Attention** · `MultiHeadSelfAttention`, `CrossModalAttentionFusion`
$$\text{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$
**Why $\sqrt{d_k}$.** If components of $q,k$ are independent zero-mean unit-variance, then
$q\!\cdot\!k=\sum_{i=1}^{d_k}q_ik_i$ has variance $d_k$. Dividing by $\sqrt{d_k}$ rescales the
logits to unit variance, preventing softmax saturation (vanishing gradients).

#### **E22 — Softmax**
$$\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$$
Used for attention weights and class probabilities; the canonical link from logits to a
probability simplex.

#### **E23 — Multi-Head Attention** · `MultiHeadSelfAttention`
$$\text{head}_h=\text{Attn}(XW_h^Q,XW_h^K,XW_h^V),\quad
\text{MHA}(X)=\big[\text{head}_1\Vert\dots\Vert\text{head}_8\big]W^O$$
with $d_k=d_v=d_\text{model}/h=512/8=64$. Heads attend to different sub-spaces in parallel.

#### **E24 — Layer Normalization** · used throughout
$$\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\hat x+\beta,\quad
\mu=\tfrac1d\sum x_i,\ \sigma^2=\tfrac1d\sum(x_i-\mu)^2$$
Normalizes across features per token; stabilizes training of deep recurrent/attention stacks.

#### **E25 — 1-D Convolution** · `TextCNN`, `PriceBranch`
$$y_c[t]=\sum_{c'=1}^{C_{in}}\sum_{j=0}^{k-1}w_{c,c'}[j]\,x_{c'}[t+j-p]+b_c$$
Cross-correlation of a length-$k$ kernel over time; extracts local $k$-gram (price/word)
patterns.

#### **E26 — Dilated Convolution & receptive field** · `PriceBranch.dilated_convs`
$$y[t]=\sum_{j=0}^{k-1}w[j]\,x[t-d\cdot j],\qquad
R = 1+\sum_{\ell}(k_\ell-1)\,d_\ell$$
With $k=3$, dilations $\{1,2,4,8\}$: $R=1+2(1+2+4+8)=31$ days — one month of context with few
layers. Derivation: each layer adds $(k-1)d_\ell$ to the span; summing telescopes the receptive
field (van den Oord et al., WaveNet 2016).

#### **E27 — GELU** · `PriceBranch`, `DualHeadOutput`
$$\text{GELU}(x)=x\,\Phi(x)=\tfrac{x}{2}\Big[1+\operatorname{erf}\!\big(\tfrac{x}{\sqrt2}\big)\Big]$$
Gaussian-gated activation: input scaled by the probability a standard normal is below it
(Hendrycks & Gimpel, 2016).

#### **E28 — LSTM cell** · `text_branch`, `price_branch`
$$\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f) & i_t&=\sigma(W_i[h_{t-1},x_t]+b_i)\\
o_t&=\sigma(W_o[h_{t-1},x_t]+b_o) & \tilde c_t&=\tanh(W_c[h_{t-1},x_t]+b_c)\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t & h_t&=o_t\odot\tanh(c_t)
\end{aligned}$$
Gated memory: forget/input/output gates regulate the constant-error carousel $c_t$, mitigating
vanishing gradients (Hochreiter & Schmidhuber, 1997). BiLSTM runs two LSTMs in opposite time
directions and concatenates: $h_t=[\overrightarrow{h_t}\Vert\overleftarrow{h_t}]$.
> *Code note:* `_init_lstm_weights` sets the forget-gate bias slice to 1.0 — a standard trick
> so gates start "open," easing early gradient flow (Jozefowicz et al., 2015).

#### **E29 — Variational (locked) Dropout** · `VariationalDropout`
$$\tilde x_t = m\odot x_t/(1-p),\quad m\sim\text{Bernoulli}(1-p)\ \text{shared}\ \forall t$$
The *same* mask is applied at every timestep (vs. fresh per step), giving a sound Bayesian
approximation for RNNs (Gal & Ghahramani, 2016). The $1/(1-p)$ keeps $\mathbb E[\tilde x]=x$
(inverted dropout).

#### **E30 — Xavier/Glorot & E31 — Orthogonal init** · `_init_weights`, `_init_lstm_weights`
$$W_{ij}\sim U\!\Big[-\sqrt{\tfrac{6}{n_{in}+n_{out}}},\ \sqrt{\tfrac{6}{n_{in}+n_{out}}}\Big]$$
Chosen so $\operatorname{Var}(W)=2/(n_{in}+n_{out})$, preserving activation/gradient variance
across layers (Glorot & Bengio, 2010). Recurrent `weight_hh` uses orthogonal $W$ ($W^\top W=I$)
so repeated multiplication neither explodes nor vanishes (Saxe et al., 2013).

#### **E36 — Dual-head outputs** · `dual_head_output.py`
$$\mathbf p=\operatorname{softmax}(W_d g(z)),\qquad \hat r = W_r g(z)$$
where $g(z)=\text{Dropout}(\text{GELU}(W_1 z))$. Classification simplex + linear regression head.

### A.3 Loss, optimization, calibration (standard forms)

#### **E37 — Cross-Entropy** · production `trainer.py`
$$\mathcal L_{CE}=-\sum_{c=1}^{3}y_c\log p_c = -\log p_{y}$$
Negative log-likelihood of the true class under the softmax model.

#### **E38 — Huber loss** · `trainer.py` `HuberLoss(delta=0.05)`
$$\mathcal L_\delta(e)=\begin{cases}\tfrac12 e^2 & |e|\le\delta\\[2pt]
\delta\big(|e|-\tfrac12\delta\big)&|e|>\delta\end{cases},\quad e=\hat r-r$$
Quadratic near zero (efficient like MSE), linear in the tails (robust like MAE to return
outliers/fat tails). Continuity & $C^1$ at $|e|=\delta$ by construction. (Huber, 1964.)

#### **E39 — Focal Loss** · `finsent/training/losses.py: FocalLoss` (γ existing; α tuned → see T1)
$$\mathcal L_{FL}=-\alpha_t(1-p_t)^{\gamma}\log p_t$$
**Derivation/intuition.** Start from CE $=-\log p_t$. Multiply by modulating factor
$(1-p_t)^\gamma$: for easy examples $p_t\to1\Rightarrow(1-p_t)^\gamma\to0$ (loss suppressed); for
hard examples $p_t\to0\Rightarrow$ factor $\to1$ (full loss). $\gamma=2$ is the canonical value
(Lin et al., 2017).

#### **E40 — Binary Cross-Entropy** · `ConfidenceCalibrationLoss`
$$\mathcal L_{BCE}=-\big[y\log\hat y+(1-y)\log(1-\hat y)\big]$$

#### **E41 — AdamW update** · optimizer
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat m_t=\tfrac{m_t}{1-\beta_1^t},\ \hat v_t=\tfrac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\eta\Big(\tfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_{t-1}\Big)$$
Adam with **decoupled** weight decay (the $\lambda\theta$ term is outside the adaptive scaling)
(Loshchilov & Hutter, 2019). Bias-correction terms $1-\beta^t$ undo the zero-initialization
of the moment estimates.

#### **E42 — Gradient clipping** · `gradient_clip_norm: 1.0`
$$g\leftarrow g\cdot\min\!\Big(1,\ \frac{\tau}{\lVert g\rVert_2}\Big)$$
Rescales the gradient if its norm exceeds $\tau$, bounding update size on loss cliffs.

#### **E43 — Temperature Scaling** · `calibration.py: TemperatureScaler`
$$\mathbf p^{\text{cal}}=\operatorname{softmax}(\mathbf z/T),\qquad
T^*=\arg\min_T \ \text{NLL}\big(\operatorname{softmax}(\mathbf z/T),y\big)$$
Single scalar $T$ optimized by LBFGS on the validation set; $T>1$ softens over-confident logits
without changing the argmax (so accuracy is invariant) (Guo et al., 2017).

#### **E44 — Expected Calibration Error (ECE)** · `_compute_ece`
$$\text{ECE}=\sum_{b=1}^{B}\frac{|B_b|}{N}\,\big|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\big|$$
Bins predictions by confidence; measures the gap between confidence and empirical accuracy.
Used as a metric and (T2) as a differentiable penalty.

#### **E45 — Cosine schedule with linear warmup** · `schedulers.py`
$$\eta_t=\begin{cases}\eta_{\max}\dfrac{t}{T_w}& t<T_w\\[8pt]
\eta_{\min}+(\eta_{\max}-\eta_{\min})\cdot\tfrac12\big(1+\cos(\pi\,t')\big)&t\ge T_w\end{cases},\quad
t'=\frac{t-T_w}{T-T_w}$$
Linear warmup avoids early instability on freshly-normalized features; cosine decay anneals
smoothly to a wide minimum (Loshchilov & Hutter, 2017).

### A.4 GAN mathematics (research augmentation, `finsent/models/gan.py`)

#### **E46 — WGAN-GP objective**
$$\min_G\max_{D\in\mathcal D_{1\text{-Lip}}}\ \underbrace{\mathbb E_{x\sim p_r}[D(x)]-\mathbb E_{z}[D(G(z))]}_{\text{Wasserstein-1 dual (Kantorovich–Rubinstein)}}\ -\ \lambda\,\mathbb E_{\hat x}\big[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2-1)^2\big]$$
**Derivation.** By Kantorovich–Rubinstein duality, the Earth-Mover distance is
$W(p_r,p_g)=\sup_{\lVert D\rVert_L\le1}\mathbb E_{p_r}[D]-\mathbb E_{p_g}[D]$. We enforce the
1-Lipschitz constraint not by weight-clipping but by penalizing the deviation of the gradient
norm from 1 (its optimal value along interpolants) (Gulrajani et al., 2017).

#### **E47 — Gradient penalty** · `compute_gradient_penalty`
$$\hat x=\epsilon x+(1-\epsilon)G(z),\ \epsilon\sim U(0,1);\qquad
\text{GP}=\lambda\,\mathbb E\big[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2-1)^2\big],\ \lambda=10$$
Sampled on random interpolations between real and fake — where the optimal critic's gradient
norm provably equals 1.

#### **E48 — LeakyReLU** · generator/critic
$$\text{LeakyReLU}(x)=\max(0.2x,\ x)$$

---

## SECTION B — TUNED EQUATIONS (modified from standard forms)

#### **T1 — Finance-weighted Focal α** · `config.yaml`, `FinSentLoss`
$$\mathcal L_{FL}=-\alpha_t(1-p_t)^{\gamma}\log p_t,\quad
\alpha=(\alpha_{\uparrow},\alpha_{\text{neu}},\alpha_{\downarrow})=(0.3,\ 0.4,\ 0.3),\ \gamma=2$$
The standard focal loss uses a single $\alpha$; here it is a **per-class prior vector** tuned to
the Up/Neutral/Down imbalance (neutral is the majority class, hence up-weighted partners), with
label smoothing 0.05 folded in.

#### **T2 — ECE as a trainable penalty (differentiable calibration)** · `ConfidenceCalibrationLoss`
$$\mathcal L_{\text{conf}}=\underbrace{\mathcal L_{BCE}(\hat q,\ \mathbb 1[\hat y=y])}_{\text{correctness supervision}}+\ w_{\text{ece}}\cdot\text{ECE},\quad w_{\text{ece}}=1$$
ECE (E44) is normally an *evaluation* metric. Here it is repurposed as an additive **training**
penalty, and the confidence target is the model's own correctness indicator — a modification of
standard calibration practice.

#### **T3 — Production multi-task loss weights** · `trainer.py`
$$\mathcal L=\mathcal L_{CE}(\text{direction})+0.5\cdot\mathcal L_{\delta=0.05}(\text{return})$$
A tuned $0.5$ task-balance and a tuned Huber threshold $\delta=0.05$ (≈ a 5% return scale).
The research path uses the alternative weighting $0.7\,\mathcal L_{\text{dir}}+0.3\,\mathcal L_{\text{conf}}$ (also tuned).

#### **T4 — Kelly criterion with safety cap & risk-appetite scaling** · `signal_generator.py`, `kelly_sizer.py`
Standard Kelly (see E-form below) is **truncated and scaled**:
$$f^\star=\frac{b\,p-q}{b},\qquad
f_{\text{used}}=\underbrace{\operatorname{clip}(f^\star,0,0.25)}_{\text{¼-Kelly cap}}\cdot\underbrace{\max(\rho,0.1)}_{\text{risk-appetite}}$$
**Derivation of base Kelly.** Maximize expected log-growth of bankroll. A bet returning $+b$ with
prob $p$ or $-1$ with prob $q=1-p$ on fraction $f$ gives growth
$G(f)=p\ln(1+bf)+q\ln(1-f)$. Setting $G'(f)=\frac{pb}{1+bf}-\frac{q}{1-f}=0$ and solving:
$pb(1-f)=q(1+bf)\Rightarrow pb-pbf=q+qbf\Rightarrow f^\star=\frac{pb-q}{b}=p-\frac{q}{b}$. ∎
The ¼ cap controls variance (full Kelly is notoriously volatile); the $\max(\rho,0.1)$ multiplier
makes sizing proportional to the user's stated risk appetite — both project-specific tunings.

#### **T5 — Reward/Risk (R:R) ratio** · `signal_generator.py`
$$b = \text{R:R} = \frac{\text{reward}}{\text{risk}}=\frac{|P_{\text{target}}-P_0|}{|P_0-P_{\text{stop}}|}$$
Standard R:R, but it is fed *directly as the Kelly odds $b$* — coupling trade geometry to bet
sizing.

#### **T6 — ATR-scaled target & stop, modulated by risk appetite** · `signal_generator.py`
$$\begin{aligned}
\text{target mult } m_T &= 2.5 + 1.5\,\rho\\
\text{stop mult } m_S &= 1.5 + 1.0\,(1-\rho)\\
P_{\text{target}}&=P_0(1+|\hat r|),\qquad
P_{\text{stop}}=P_0 - m_S\cdot\text{ATR}_{14}\quad(\text{long})
\end{aligned}$$
A tuned generalization of the textbook "stop = entry − k·ATR." Higher risk appetite $\rho$
widens the target and tightens the stop (more aggressive); the multipliers $2.5/1.5/1.0$ are
hand-tuned.

#### **T7 — Annualized risk-adjusted ratios with per-trade RF conversion** · `risk_engine.py`
$$\text{Sharpe}=\frac{\bar r - r_f/252}{\operatorname{std}(r)}\sqrt{252},\qquad
\text{Sortino}=\frac{\bar r - r_f/252}{\operatorname{std}(r\,|\,r<0)}\sqrt{252}$$
$$\text{Calmar}=\frac{252\,\bar r}{|\text{MaxDD}|}$$
Standard ratios, **tuned** by converting the annual risk-free rate to a per-trading-day hurdle
$r_f/252$ before annualizing — the implementation detail that makes the daily-return Sharpe
consistent. Sortino replaces total volatility with **downside** deviation only.

#### **T8 — Intensifier-boosted lexicon sentiment** · `news_sentiment_engine.py`
$$s_{\text{raw}}=\frac{n_+ - n_-}{n_+ + n_-},\qquad
s=\operatorname{clip}\big(s_{\text{raw}}\cdot(1+0.2\min(n_{\text{int}},3)),\,-1,\,1\big)$$
A Loughran–McDonald-style polarity count (existing idea) **tuned** with an intensifier
amplification factor (each of up to 3 intensifier words adds 20% magnitude) and a confidence
$c=\min(0.4+0.1\,(n_++n_-),\,0.85)$.

#### **T9 — Analytical long-only max-Sharpe tilt** · `portfolio_optimizer.py`
$$w \propto \big[\Sigma^{-1}(\mu-r_f/252\,\mathbf 1)\big]_+,\qquad w\leftarrow w/\textstyle\sum w$$
**Derivation of the unconstrained tangency portfolio.** Maximizing the Sharpe ratio
$\frac{w^\top\mu - r_f}{\sqrt{w^\top\Sigma w}}$ yields the first-order condition
$w^\star\propto\Sigma^{-1}(\mu-r_f\mathbf1)$ (the classic Markowitz tangency solution). This code
**tunes** it for a long-only book by clipping negatives $[\cdot]_+$ and renormalizing, plus a
ridge term $\Sigma+10^{-8}I$ for numerical stability — an approximation, not the exact QP.

---

## SECTION C — CUSTOM EQUATIONS (made for this project)

#### **C1 — Triple-barrier-lite direction labeling** · `training/dataset.py`
$$r^{\text{fwd}}_t=\frac{C_{t+H}}{C_t}-1,\qquad
y_t=\begin{cases}
0\ (\text{UP}) & r^{\text{fwd}}_t > \tau_{\uparrow}\\
2\ (\text{DOWN}) & r^{\text{fwd}}_t < \tau_{\downarrow}\\
1\ (\text{NEUTRAL}) & \text{otherwise}
\end{cases}$$
Custom three-way labeling over horizon $H$ with a neutral dead-band $[\tau_\downarrow,\tau_\uparrow]$
(config `neutral_threshold = 0.005`). It deliberately abstains on small moves so the classifier
isn't penalized for noise — a project-specific simplification of López de Prado's triple-barrier
method. The same $r^{\text{fwd}}$ is the regression target $r$.

#### **C2 — Cross-Modal Gated Attention Fusion (the core innovation)** · `cross_modal_fusion.py`
Sentiment vector $s\in\mathbb R^{512}$ is the **query**; the price sequence $P\in\mathbb R^{30\times512}$
is **key/value**:
$$q = W_q\,\text{LN}(s),\quad K=W_k\,\text{LN}(P),\quad V=W_v\,\text{LN}(P)$$
$$\boldsymbol\alpha=\operatorname{softmax}\!\Big(\frac{qK^\top}{\sqrt{d_k}}\Big)\in\mathbb R^{1\times30},\qquad
\text{ctx}=W_o(\boldsymbol\alpha V)$$
$\boldsymbol\alpha$ are **interpretable** weights: which of the past 30 days the current sentiment
"attends" to. Custom because the *cross-modal* direction (text→price) and its surfacing as an
explanation overlay are bespoke.

#### **C3 — Learned modal gate** · `cross_modal_fusion.py`
$$g=\sigma\big(W_g[s\,\Vert\,\text{ctx}]+b_g\big)\in(0,1)^{512},\qquad
\boxed{f = g\odot\text{ctx} + (1-g)\odot s}$$
A learned, element-wise convex blend of the price-attended context and the raw sentiment. The
network *learns when sentiment dominates* (e.g., earnings/news days, $g\to0$) **vs.** when price
action dominates (technical breakouts, $g\to1$). This adaptive gate is the project's signature
equation.

#### **C4 — Confidence-as-correctness supervision** · `ConfidenceCalibrationLoss`
$$y^{\text{conf}}_i = \mathbb 1\big[\arg\max_c \text{logit}_{i,c} = y_i\big],\qquad
\hat q_i \xrightarrow{\text{BCE}} y^{\text{conf}}_i$$
Custom training signal: the confidence head is supervised to predict *its own correctness*, so
that emitted confidence is directly usable as Kelly's $p$ (see T4). Couples calibration to
position sizing by construction.

#### **C5 — Combined FinSent objective** · `FinSentLoss`
$$\mathcal L_{\text{FinSent}} = w_d\,\mathcal L_{FL}^{(\alpha,\gamma)} + w_c\big(\mathcal L_{BCE}+w_{\text{ece}}\text{ECE}\big),
\quad (w_d,w_c)=(0.7,0.3)$$
A bespoke multi-task scalarization fusing finance-weighted focal classification (T1) with the
custom calibration loss (C4/T2).

#### **C6 — Discrete signal mapping from $P_{\text{up}}$** · `signal_generator.py`
$$\text{signal}(P_{\uparrow})=\begin{cases}
\text{STRONG BUY} & P_{\uparrow}\ge0.70\\
\text{BUY} & 0.55\le P_{\uparrow}<0.70\\
\text{HOLD} & 0.40\le P_{\uparrow}<0.55\\
\text{SELL} & 0.25\le P_{\uparrow}<0.40\\
\text{STRONG SELL} & P_{\uparrow}<0.25
\end{cases}$$
Custom thresholding of the calibrated up-probability into a 5-level action that drives the
BUY/SELL/HOLD markers and entry/exit lines on the candlestick chart.

#### **C7 — Position quantity & capital plan** · `signal_generator.py`, `allocation_engine.py`
$$\text{Capital}_i = C\cdot w_i,\qquad
q_i=\max\!\big(1,\big\lfloor \tfrac{C\cdot f_{\text{used},i}}{P_{0,i}}\big\rfloor\big),\qquad
\text{Cost}_i=q_i P_{0,i}(1+\phi)$$
$$\text{Deployed}=\sum_i\text{Cost}_i,\quad \text{Cash}=C-\text{Deployed},\quad
\text{Utilization}=\frac{\text{Deployed}}{C}$$
The final per-stock share count, deployed capital, holding (cash) amount, and broker-fee-adjusted
($\phi$) cost — i.e., the concrete numbers in the "Analyze" summary.

#### **C8 — Composite technical score** · `signal_generator.py`
$$\text{TechScore}=\operatorname{clip}\Big(50 + \Delta_{\text{RSI}} + \Delta_{\text{MACD}} + \Delta_{\text{Golden}} + \Delta_{\text{BB}},\ 0,\ 100\Big)$$
with rule increments $\Delta_{\text{RSI}}=\{+20\text{ if }RSI<30;\,+10\text{ if }30\le RSI\le70\}$,
$\Delta_{\text{MACD}}=+15$ if MACD > Signal, $\Delta_{\text{Golden}}=+10$ if 50-SMA > 200-SMA,
$\Delta_{\text{BB}}=+15$ if BB-position < 0.2. A custom 0–100 heuristic blended into the displayed
score and reasoning.

#### **C9 — Custom regime classifier** · `regime_detector.py`
$$\text{Regime}=\begin{cases}
\text{VOLATILE}& \sigma^{\text{ann}}>0.25\\
\text{BULL}& C>\text{SMA}_{50}>\text{SMA}_{200}\ \wedge\ r^{20d}>0\\
\text{BEAR}& C<\text{SMA}_{50}<\text{SMA}_{200}\ \wedge\ r^{20d}<0\\
\text{TRANSITIONAL}&\text{otherwise}\end{cases}$$
with a volatility z-score $z_\sigma=\frac{\sigma^{\text{ann}}-\sigma^{\text{ann}}_{60}}{\sigma^{\text{ann}}_{60}+\epsilon}$
and confidences such as $0.6+\min(5\,r^{20d},0.3)$. The 5-state GAN-side variant
(`gan.py: RegimeDetector`) classifies via $z_\sigma$ thresholds $\{2.0,1.0,-1.0\}$ into
{Bull, Bear, Crisis, HighVol, LowVol}. Both rule sets are bespoke.

#### **C10 — Sentiment aggregation & normalization** · `news_sentiment_engine.py`
$$\bar s=\frac1n\sum_i s_i,\quad \text{label}=\operatorname{sgn}_{0.15}(\bar s),\quad
\tilde s = \tfrac{(\bar s+1)}{2}\cdot100\ \in[0,100]$$
with positive/negative percentages $\frac{\#\{s_i:\text{label}=+\}}{n}$. Maps the news layer into
a 0–100 sentiment gauge shown on the dashboard and into the model's text-token bucket.

#### **C11 — Sentiment→token bucket (deterministic inference bridge)** · `api/routes/analysis.py`
$$\text{bucket}=\operatorname{clip}\big(\lfloor 300\cdot\operatorname{clip}(s,-1,1)+500\rfloor,\ 100,\ 999\big)$$
Custom deterministic encoding that injects the scalar sentiment into the text-token stream at
inference (avoiding random-noise drift) so the same inputs yield reproducible predictions.

#### **C12 — Candlestick-pattern indicator encodings** · `technical_indicators.py`
$$\text{Doji}=\mathbb 1\!\Big[\tfrac{|C-O|}{H-L}<0.1\Big],\quad
\text{Hammer}=\mathbb 1\big[\text{lower-shadow}\ge2|C-O|\ \wedge\ \text{upper-shadow}<|C-O|\big]$$
$$\text{Engulf}=\mathbb 1\big[C>O\ \wedge\ C_{-1}<O_{-1}\ \wedge\ C>O_{-1}\ \wedge\ O<C_{-1}\big]$$
Custom binary feature encodings of classic candlestick patterns, fed as model features and used
to train "all the most possible patterns."

---

## SECTION D — PORTFOLIO & RISK EQUATIONS (existing, used as-is)

#### **E49 — Portfolio return & variance** · `portfolio_optimizer.py`
$$\mu_p = w^\top\mu\cdot252,\qquad \sigma_p=\sqrt{w^\top\Sigma w\cdot252}$$

#### **E50 — Global minimum-variance portfolio**
$$w_{\text{mv}}=\frac{\Sigma^{-1}\mathbf 1}{\mathbf 1^\top\Sigma^{-1}\mathbf 1}$$
**Derivation.** Minimize $w^\top\Sigma w$ s.t. $\mathbf 1^\top w=1$. Lagrangian
$\mathcal L=w^\top\Sigma w-\lambda(\mathbf1^\top w-1)$; $\nabla_w=2\Sigma w-\lambda\mathbf1=0
\Rightarrow w=\tfrac{\lambda}{2}\Sigma^{-1}\mathbf1$; normalize by the constraint. ∎ (Code clips
to long-only and renormalizes.)

#### **E51 — Risk-parity (equal risk contribution)** · iterative
$$\text{RC}_i = w_i\,\frac{(\Sigma w)_i}{\sqrt{w^\top\Sigma w}},\qquad
\text{target: } \text{RC}_i=\frac{\sigma_p}{n}\ \forall i$$
Solved by fixed-point reweighting $w_i\leftarrow w_i\cdot\frac{\sigma_p/n}{\text{RC}_i}$. Each asset
contributes equal risk; marginal risk contribution $\partial\sigma_p/\partial w_i=(\Sigma w)_i/\sigma_p$.

#### **E52 — Historical VaR** · `risk_engine.py`
$$\text{VaR}_\alpha = -\,\text{Percentile}_{1-\alpha}\big(\{r_t\}\big)$$

#### **E53 — Parametric (Gaussian) VaR**
$$\text{VaR}_\alpha = -\big(\bar r + z_{1-\alpha}\,\sigma_r\big),\quad z_{1-\alpha}=\Phi^{-1}(1-\alpha)$$

#### **E54 — Conditional VaR (Expected Shortfall)**
$$\text{CVaR}_\alpha = -\,\mathbb E\big[r\mid r\le -\text{VaR}_\alpha\big]$$
The mean loss in the tail beyond VaR; a coherent risk measure (sub-additive).

#### **E55 — Maximum Drawdown** · `risk_engine.py`
$$\text{Peak}_t=\max_{k\le t}E_k,\qquad \text{MaxDD}=\min_t\frac{E_t-\text{Peak}_t}{\text{Peak}_t}$$
Largest peak-to-trough equity decline along the curve $E$.

#### **E56 — Win rate** $=\frac{1}{N}\sum_t \mathbb 1[r_t>0]$.

---

## PART IV — EQUATION INDEX (paper-ready table)

| # | Equation | Category | Source file |
|---|----------|----------|-------------|
| E1–E2 | EMA / Wilder RMA | Existing | technical_indicators.py |
| E3 | RSI | Existing | technical_indicators.py |
| E4 | MACD | Existing | technical_indicators.py |
| E5–E6 | SMA / Bollinger | Existing | technical_indicators.py |
| E7 | ATR | Existing | technical_indicators.py |
| E8–E10 | OBV / VWAP / Stochastic | Existing | technical_indicators.py |
| E11–E12 | Log return / Ann. vol | Existing | technical_indicators.py |
| E19 | Causal z-score | Existing | dataset.py / analysis.py |
| E20–E23 | PosEnc / Attention / MHA | Existing | text_branch.py |
| E24–E27 | LayerNorm / Conv / Dilated / GELU | Existing | price_branch.py, text_branch.py |
| E28–E29 | LSTM / Variational dropout | Existing | text_branch.py |
| E30–E31 | Xavier / Orthogonal init | Existing | finsentnet_core.py |
| E36 | Dual-head | Existing | dual_head_output.py |
| E37–E38 | CE / Huber | Existing | trainer.py |
| E39–E40 | Focal / BCE | Existing | losses.py |
| E41–E42 | AdamW / Grad-clip | Existing | trainer.py |
| E43–E45 | Temp scaling / ECE / Cosine | Existing | calibration.py, schedulers.py |
| E46–E48 | WGAN-GP / GP / LeakyReLU | Existing | gan.py |
| E49–E56 | Portfolio & risk metrics | Existing | portfolio_optimizer.py, risk_engine.py |
| T1 | Finance-weighted focal α | Tuned | losses.py / config.yaml |
| T2 | ECE as training penalty | Tuned | losses.py |
| T3 | Multi-task loss weights | Tuned | trainer.py |
| T4 | Capped + risk-scaled Kelly | Tuned | signal_generator.py, kelly_sizer.py |
| T5 | R:R as Kelly odds | Tuned | signal_generator.py |
| T6 | Risk-modulated ATR stop/target | Tuned | signal_generator.py |
| T7 | Per-day-RF annualized ratios | Tuned | risk_engine.py |
| T8 | Intensifier-boosted lexicon | Tuned | news_sentiment_engine.py |
| T9 | Long-only max-Sharpe tilt | Tuned | portfolio_optimizer.py |
| C1 | Triple-barrier-lite labels | Custom | dataset.py |
| C2 | Cross-modal attention | Custom | cross_modal_fusion.py |
| C3 | Learned modal gate | Custom | cross_modal_fusion.py |
| C4 | Confidence-as-correctness | Custom | losses.py |
| C5 | Combined FinSent objective | Custom | losses.py |
| C6 | Signal thresholds | Custom | signal_generator.py |
| C7 | Capital/quantity plan | Custom | signal_generator.py, allocation_engine.py |
| C8 | Technical score | Custom | signal_generator.py |
| C9 | Regime classifier | Custom | regime_detector.py, gan.py |
| C10 | Sentiment aggregation | Custom | news_sentiment_engine.py |
| C11 | Sentiment→token bucket | Custom | analysis.py |
| C12 | Candlestick encodings | Custom | technical_indicators.py |

---

# PART V — MODEL DESIGN DIAGRAMS (paper-ready TikZ)

All figures are authored in **TikZ** — i.e. they *are* LaTeX, so they drop straight into a
paper with vector quality (no raster blur, fonts match the body text). A standalone,
compile-ready version of every figure lives in **`finsent_diagrams.tex`** (compile on Overleaf
with pdfLaTeX, or `pdflatex finsent_diagrams.tex`). Each figure is wrapped in `\resizebox` so it
can never overflow the column.

**Required preamble** (already in `finsent_diagrams.tex`; copy into your paper's preamble once):

```latex
\usepackage{graphicx} % for \resizebox
\usepackage{tikz}
\usepackage{amsmath}
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds,shapes.geometric,shapes.misc}

\definecolor{cInput}{HTML}{455A64}\definecolor{cText}{HTML}{1565C0}
\definecolor{cPrice}{HTML}{E65100}\definecolor{cFusion}{HTML}{6A1B9A}
\definecolor{cOut}{HTML}{00838F}\definecolor{cTrade}{HTML}{2E7D32}
\definecolor{cRisk}{HTML}{C62828}\definecolor{cGan}{HTML}{AD1457}
\tikzset{
  base/.style={rounded corners=2pt,draw,align=center,font=\small,minimum height=8mm,inner sep=4pt,line width=0.5pt},
  io/.style={base,fill=cInput!12,draw=cInput}, tnode/.style={base,fill=cText!12,draw=cText},
  pnode/.style={base,fill=cPrice!14,draw=cPrice}, fnode/.style={base,fill=cFusion!12,draw=cFusion},
  onode/.style={base,fill=cOut!14,draw=cOut}, trnode/.style={base,fill=cTrade!12,draw=cTrade},
  rnode/.style={base,fill=cRisk!12,draw=cRisk}, gnode/.style={base,fill=cGan!12,draw=cGan},
  group/.style={rounded corners=4pt,draw,dashed,line width=0.6pt,inner sep=6pt},
  arr/.style={-{Latex[length=2.2mm]},line width=0.7pt}, lbl/.style={font=\scriptsize\itshape,fill=white,inner sep=1pt}}
```

---

## Figure 1 — Whole-System Architecture

**ASCII preview (layout):**
```
 [User Inputs C,ρ,H]   [News/Headlines]   [Price Feed OHLCV]
        |                     |                    |
        |              [TEXT BRANCH]      [PRICE BRANCH]<-[35 Indicators]
        |                     |                    |
        |              s∈R^512            P∈R^30x512
        |                     \                  /
        |               [CROSS-MODAL GATED FUSION  f=g·ctx+(1-g)·s]
        |                          |
        |               [DUAL-HEAD: P(↑/–/↓)  |  r̂]
        |                          |
        └────C,ρ,H────►[SIGNAL GEN]→[PORTFOLIO OPT]→[RISK ENGINE]
                              \          |           /
                          [ FINAL INVESTMENT PLAN (Analyze) ]
```

```latex
\begin{figure}[t]\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[node distance=7mm and 12mm]
\node[io] (user)  {\textbf{User Inputs}\\ capital $C$, currency,\\ horizon $H$, risk $\rho$};
\node[io, right=24mm of user] (news) {\textbf{News / Headlines}\\ scraped + pulled\\ (multi-source)};
\node[io, right=12mm of news] (px)  {\textbf{Price Feed}\\ OHLCV (live)};
\node[tnode, below=of news] (tb) {\textbf{TEXT BRANCH}\\ FinBERT $\to$ TextCNN $\to$\\ 3$\times$BiLSTM $\to$ Self-Attn};
\node[pnode, below=of px] (pb) {\textbf{PRICE BRANCH}\\ Multi-scale Conv1D $\to$\\ Dilated TCN $\to$ 2$\times$LSTM};
\node[io, left=12mm of pb, yshift=-2mm] (ti) {\textbf{35 Technical}\\ \textbf{Indicators}\\ RSI, MACD, BB,\\ ATR, OBV, \dots};
\node[tnode, below=of tb] (svec) {sentiment vector\\ $s \in \mathbb{R}^{512}$};
\node[pnode, below=of pb] (pvec) {price sequence\\ $P \in \mathbb{R}^{30\times512}$};
\node[fnode, below=14mm of $(svec)!0.5!(pvec)$] (fus) {\textbf{CROSS-MODAL GATED FUSION}\\ $Q{=}s,\;K{=}V{=}P$\quad $f = g\odot\mathrm{ctx} + (1{-}g)\odot s$};
\node[onode, below=of fus] (dh) {\textbf{DUAL-HEAD OUTPUT}\\ Direction softmax $P(\!\uparrow\!),P(\!-\!),P(\!\downarrow\!)$\;|\; Return regressor $\hat r$};
\node[trnode, below=of dh] (sig) {\textbf{SIGNAL GENERATOR}\\ STRONG BUY\,$\dots$\,STRONG SELL\\ ATR target/stop, R:R, Kelly size};
\node[trnode, right=20mm of sig] (port) {\textbf{PORTFOLIO OPTIMIZER}\\ max-Sharpe / min-var /\\ risk-parity weights $w$};
\node[rnode, right=12mm of port] (risk) {\textbf{RISK ENGINE}\\ VaR, CVaR, MaxDD,\\ Sharpe, Sortino};
\node[io, below=of sig, xshift=28mm, fill=cTrade!22, draw=cTrade] (plan) {\textbf{FINAL INVESTMENT PLAN} \;(\textit{Analyze} button)\\ per-stock shares $q_i$, capital, holding cash, entry/target/stop,\\ BUY/SELL/HOLD \%, confidence, Kelly $f$, expected return \& Sharpe};
\draw[arr] (news)--(tb); \draw[arr] (px)--(pb); \draw[arr] (ti)--(pb);
\draw[arr] (tb)--(svec); \draw[arr] (pb)--(pvec); \draw[arr] (svec)--(fus); \draw[arr] (pvec)--(fus);
\draw[arr] (fus)--(dh); \draw[arr] (dh)--(sig); \draw[arr] (sig)--(port); \draw[arr] (port)--(risk);
\draw[arr] (sig)--(plan); \draw[arr] (port)|-(plan); \draw[arr] (risk)|-(plan);
\draw[arr] (user.south) to[out=-90,in=180] node[lbl,pos=0.6]{$C,\rho,H$} (sig.west);
\begin{scope}[on background layer]
  \node[group, draw=cText, fit=(tb)(svec)] {};
  \node[group, draw=cPrice, fit=(pb)(pvec)] {};
\end{scope}
\end{tikzpicture}}
\caption{FinSentNet end-to-end architecture.}\label{fig:arch}
\end{figure}
```

---

## Figure 2 — Text Branch (6 layers)

**ASCII preview:**
```
[L1 WordPiece tokens] → [L2 FinBERT 768-d] → E∈R^{L×768}
                                  ├──────────────┐
                          [L3 TextCNN {2,3,4,5}] [L4 3×BiLSTM 512]
                                  └──────┬───────┘
                          [inject: h0 ← h0 + CNN(E)]
                          [L5a Positional Encoding]
                          [L5b 8-Head Self-Attention]
                          [L6 s = LN(CLS + MeanPool) ∈ R^512]
```

```latex
\begin{figure}[t]\centering
\resizebox{0.92\textwidth}{!}{%
\begin{tikzpicture}[node distance=6mm]
\node[io] (tok) {\textbf{L1: WordPiece Tokens}\\ news headline $\to$ ids (seq $\le 512$)};
\node[tnode, below=of tok] (emb) {\textbf{L2: FinBERT Embeddings}\\ contextual $768$-d, last 2 layers fine-tuned};
\node[tnode, below=of emb] (split) {token embeddings $E\in\mathbb{R}^{L\times768}$};
\node[tnode, below left=10mm and 6mm of split] (cnn) {\textbf{L3: TextCNN}\\ Conv1D kernels $\{2,3,4,5\}$\\ 128 filters $\to$ maxpool $\to$ $512$-d};
\node[tnode, below right=10mm and 6mm of split] (lstm) {\textbf{L4: 3$\times$ BiLSTM}\\ hidden 256/dir $\to 512$\\ variational dropout 0.3};
\node[tnode, below=22mm of split] (inject) {CLS-stream inject:\\ $h_0 \leftarrow h_0 + \mathrm{CNN}(E)$};
\node[tnode, below=of inject] (pos) {\textbf{L5a: Positional Encoding}\\ $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$};
\node[tnode, below=of pos] (attn) {\textbf{L5b: 8-Head Self-Attention}\\ $\mathrm{softmax}(QK^\top/\sqrt{d_k})V,\;d_k{=}64$};
\node[onode, below=of attn] (pool) {\textbf{L6: Sentence Vector}\\ $s=\mathrm{LayerNorm}(\mathrm{CLS}+\mathrm{MeanPool})\in\mathbb{R}^{512}$};
\draw[arr] (tok)--(emb); \draw[arr] (emb)--(split);
\draw[arr] (split.south) -- ++(0,-3mm) -| (cnn.north);
\draw[arr] (split.south) -- ++(0,-3mm) -| (lstm.north);
\draw[arr] (cnn.south) |- (inject.west); \draw[arr] (lstm.south) |- (inject.east);
\draw[arr] (inject)--(pos); \draw[arr] (pos)--(attn); \draw[arr] (attn)--(pool);
\end{tikzpicture}}
\caption{Text branch encoder.}\label{fig:text}
\end{figure}
```

---

## Figure 3 — Price Branch (5 layers)

**ASCII preview:**
```
[X∈R^{30×20}] ─┬─[Conv1D k=3]─┐
               ├─[Conv1D k=7]─┼─[Concat+BN 192]─[Dilated TCN {1,2,4,8}]
               └─[Conv1D k=15]┘        │(+residual 1×1)
                            [2×LSTM 256]
                            [Proj→LN→GELU 512]
                            [context p + sequence P]
```

```latex
\begin{figure}[t]\centering
\resizebox{0.95\textwidth}{!}{%
\begin{tikzpicture}[node distance=6mm and 10mm]
\node[io] (in) {\textbf{Input}\\ window $X\in\mathbb{R}^{30\times20}$\\ (causal z-score)};
\node[pnode, right=16mm of in, yshift=9mm]  (c3)  {Conv1D $k{=}3$\\ (3-day)};
\node[pnode, right=16mm of in]              (c7)  {Conv1D $k{=}7$\\ (weekly)};
\node[pnode, right=16mm of in, yshift=-9mm] (c15) {Conv1D $k{=}15$\\ (monthly)};
\node[pnode, right=14mm of c7] (cat) {\textbf{Concat + BN}\\ $192$-d multi-scale};
\node[pnode, right=12mm of cat] (dil) {\textbf{Dilated TCN}\\ $d\in\{1,2,4,8\}$\\ recept. field $\approx 31$d};
\node[pnode, below=12mm of dil] (res) {residual $1{\times}1$ conv\\ skip connection};
\node[pnode, below=14mm of cat] (lstm) {\textbf{2$\times$ LSTM}\\ hidden 256\\ momentum / mean-rev};
\node[onode, left=14mm of lstm] (proj) {\textbf{Projection}\\ Linear$\to$LN$\to$GELU\\ to $512$-d};
\node[onode, below=8mm of proj] (out) {context $p\in\mathbb{R}^{512}$\\ + sequence $P\in\mathbb{R}^{30\times512}$};
\draw[arr] (in.east)--(c3.west); \draw[arr] (in.east)--(c7.west); \draw[arr] (in.east)--(c15.west);
\draw[arr] (c3)--(cat); \draw[arr] (c7)--(cat); \draw[arr] (c15)--(cat); \draw[arr] (cat)--(dil);
\draw[arr] (cat.south) to[out=-90,in=180] (res.west);
\draw[arr] (res.east) to[out=0,in=-90] node[lbl]{$+$} (dil.south);
\draw[arr] (dil.south) -- ++(0,-6mm) -| (lstm.north);
\draw[arr] (lstm)--(proj); \draw[arr] (proj)--(out);
\end{tikzpicture}}
\caption{Price branch encoder.}\label{fig:price}
\end{figure}
```

---

## Figure 4 — Cross-Modal Gated Fusion (core innovation)

**ASCII preview:**
```
 s∈R^512 ─►[Q=Wq·LN(s)]─┐
                         ├─►[Cross-Attn  α=softmax(QKᵀ/√dk); ctx=Wo(αV)]─►[Gate g=σ(Wg[s‖ctx])]
 P∈R^{30×512}─►[K,V]─────┘                                                        │
            └──────────────────── s ────────────────────────► f = g·ctx + (1−g)·s
```

```latex
\begin{figure}[t]\centering
\resizebox{0.9\textwidth}{!}{%
\begin{tikzpicture}[node distance=8mm and 14mm]
\node[tnode] (s) {sentiment $s\in\mathbb{R}^{512}$\\ \scriptsize ``what is the market feeling?''};
\node[pnode, below=20mm of s] (P) {price seq $P\in\mathbb{R}^{30\times512}$\\ \scriptsize ``what patterns exist?''};
\node[fnode, right=of s] (q) {$Q=W_q\,\mathrm{LN}(s)$};
\node[fnode, right=of P] (kv){$K=W_k\mathrm{LN}(P)$\\ $V=W_v\mathrm{LN}(P)$};
\node[fnode, right=16mm of $(q)!0.5!(kv)$] (att) {\textbf{Cross-Attention}\\ $\boldsymbol\alpha=\mathrm{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)$\\ $\mathrm{ctx}=W_o(\boldsymbol\alpha V)$};
\node[fnode, right=of att] (gate) {\textbf{Learned Gate}\\ $g=\sigma\big(W_g[s\,\Vert\,\mathrm{ctx}]+b_g\big)$};
\node[onode, below=16mm of gate] (fused) {\textbf{Fused vector}\\ $f=g\odot\mathrm{ctx}+(1{-}g)\odot s$};
\draw[arr] (s)--(q); \draw[arr] (P)--(kv); \draw[arr] (q)--(att); \draw[arr] (kv)--(att);
\draw[arr] (att)-- node[lbl]{ctx} (gate); \draw[arr] (gate)--(fused);
\draw[arr] (s.south) to[out=-90,in=180] node[lbl,pos=0.7]{$s$} (fused.west);
\end{tikzpicture}}
\caption{Cross-modal gated attention fusion (the core innovation).}\label{fig:fusion}
\end{figure}
```

---

## Figure 5 — Dual-Head Output

```latex
\begin{figure}[t]\centering
\resizebox{0.78\textwidth}{!}{%
\begin{tikzpicture}[node distance=8mm and 16mm]
\node[fnode] (f) {fused $f\in\mathbb{R}^{512}$};
\node[onode, above right=6mm and 16mm of f] (dch) {\textbf{Head A: Direction}\\ FC$\to$GELU$\to$Drop(0.4)$\to$FC(3)};
\node[onode, below right=6mm and 16mm of f] (rch) {\textbf{Head B: Return}\\ FC$\to$GELU$\to$Drop(0.3)$\to$FC(1)};
\node[trnode, right=of dch] (probs) {$\mathrm{softmax}\Rightarrow$\\ $P(\!\uparrow\!),\,P(\!-\!),\,P(\!\downarrow\!)$};
\node[trnode, right=of rch] (ret) {predicted return $\hat r$};
\draw[arr] (f)--(dch); \draw[arr] (f)--(rch); \draw[arr] (dch)--(probs); \draw[arr] (rch)--(ret);
\end{tikzpicture}}
\caption{Dual-head output.}\label{fig:head}
\end{figure}
```

---

## Figure 6 — WGAN-GP Crisis Augmentation

```latex
\begin{figure}[t]\centering
\resizebox{0.92\textwidth}{!}{%
\begin{tikzpicture}[node distance=8mm and 14mm]
\node[io] (z) {noise $z\sim\mathcal{N}(0,I)$\\ $+$ regime $r$ embed};
\node[gnode, right=of z] (G) {\textbf{Generator} $G(z,r)$\\ MLP $\to$ Tanh};
\node[gnode, right=of G] (fake) {synthetic window\\ $\tilde x\in\mathbb{R}^{30\times15}$};
\node[io, below=16mm of z] (real) {real crisis window $x$};
\node[gnode, right=18mm of $(fake)!0.5!(real)$] (C) {\textbf{Critic} $D(\cdot,r)$\\ LayerNorm MLP\\ (no sigmoid)};
\node[rnode, right=of C] (loss) {\textbf{WGAN-GP loss}\\ $\mathbb{E}[D(x)]-\mathbb{E}[D(\tilde x)]$\\ $+\lambda\,\mathbb{E}\big[(\Vert\nabla D(\hat x)\Vert_2-1)^2\big]$};
\draw[arr] (z)--(G); \draw[arr] (G)--(fake); \draw[arr] (fake)--(C); \draw[arr] (real)--(C); \draw[arr] (C)--(loss);
\draw[arr] (loss.south) to[out=-90,in=-90] node[lbl,pos=0.5]{$\min_G\max_D$} (G.south);
\end{tikzpicture}}
\caption{WGAN-GP regime-conditioned crisis augmentation.}\label{fig:gan}
\end{figure}
```

---

## Figure 7 — Training Pipeline

```latex
\begin{figure}[t]\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[node distance=6mm and 12mm]
\node[io] (data) {OHLCV + news\\ history};
\node[io, right=of data] (win) {window $W{=}30$,\\ causal z-score};
\node[io, right=of win] (lab) {\textbf{labels}\\ $r^{\mathrm{fwd}}{=}\tfrac{C_{t+H}}{C_t}{-}1$\\ UP/NEU/DOWN + $\hat r$};
\node[fnode, right=of lab] (model) {FinSentNet\\ forward pass};
\node[rnode, below=14mm of model] (loss) {\textbf{Loss}\\ \emph{prod:} CE $+\,0.5\,$Huber\\ \emph{research:} $0.7\,$Focal$+0.3\,$Calib};
\node[trnode, left=of loss] (opt) {\textbf{AdamW}\\ cosine-warmup LR\\ grad-clip $\Vert g\Vert{\le}1$};
\node[trnode, left=of opt] (es) {\textbf{Early stop}\\ monitor val Sharpe\\ patience 15};
\node[onode, left=of es] (cal) {\textbf{Temp. scaling}\\ $T^\star=\arg\min$ NLL\\ (LBFGS, post-hoc)};
\draw[arr] (data)--(win); \draw[arr] (win)--(lab); \draw[arr] (lab)--(model);
\draw[arr] (model)--(loss); \draw[arr] (loss)--(opt); \draw[arr] (opt)--(es); \draw[arr] (es)--(cal);
\draw[arr] (opt.north) to[out=120,in=-60] node[lbl]{update $\theta$} (model.south);
\end{tikzpicture}}
\caption{Training pipeline.}\label{fig:train}
\end{figure}
```

---

## Figure 8 — Inference → Decision → Final Summary

```latex
\begin{figure}[t]\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[node distance=6mm and 11mm]
\node[io] (u) {\textbf{User}\\ $C,\rho,H$,\\ market, stocks};
\node[io, right=of u] (fetch) {fetch OHLCV\\ + 35 indicators};
\node[io, right=of fetch] (sent) {news scrape\\ + sentiment $s$};
\node[fnode, right=of sent] (infer) {batched\\ model inference};
\node[trnode, below=14mm of infer] (map) {\textbf{Signal map}\\ $P(\!\uparrow\!)\Rightarrow$\\ STRONG BUY$\dots$SELL};
\node[trnode, left=of map] (size) {\textbf{Kelly sizing}\\ $f^\star=\tfrac{bp-q}{b}$\\ cap $\tfrac14$, $\times\rho$};
\node[trnode, left=of size] (alloc) {\textbf{Allocation}\\ weights $w_i$,\\ shares $q_i$, fees};
\node[rnode, left=of alloc] (riskr) {\textbf{Risk report}\\ VaR/CVaR,\\ Sharpe, MaxDD};
\node[io, below=12mm of alloc, fill=cTrade!22, draw=cTrade, text width=85mm] (sum) {\textbf{FINAL SUMMARY CARD} --- per stock: shares $q_i$, entry / target / stop, capital deployed, holding cash, BUY/SELL/HOLD \%, confidence, $f$; portfolio: total deployed, utilization, expected return, volatility, Sharpe.};
\draw[arr] (u)--(fetch); \draw[arr] (fetch)--(sent); \draw[arr] (sent)--(infer);
\draw[arr] (infer)--(map); \draw[arr] (map)--(size); \draw[arr] (size)--(alloc); \draw[arr] (alloc)--(riskr);
\draw[arr] (alloc)--(sum); \draw[arr] (size.south)|-(sum.east); \draw[arr] (riskr.south)|-(sum.west);
\end{tikzpicture}}
\caption{Inference-to-decision flow producing the final investment summary.}\label{fig:flow}
\end{figure}
```

---

### How to render to images (for a non-LaTeX paper, e.g. Word)
1. Open `finsent_diagrams.tex` on [Overleaf](https://overleaf.com) → compile → download PDF (each figure on its own page).
2. To get PNG/SVG per figure: change the class to `\documentclass[border=6pt]{standalone}`, put **one** `tikzpicture` per file, compile, then `pdftoppm -png -r 600 fig.pdf fig` (600 dpi) or `pdf2svg fig.pdf fig.svg`.
3. In a LaTeX paper, you don't need images at all — paste the `\begin{figure}…\end{figure}` blocks directly.

