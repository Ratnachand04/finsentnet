# FinSent Model Functional Flow

This document links the main runtime path across backend model inference and frontend consumption.

## 1. Backend bootstrap
- Runtime env bootstrap: [_bootstrap_runtime_env](finsentnet_pro/backend/api/main.py#L25)
- Device selection: [_select_torch_device](finsentnet_pro/backend/api/main.py#L61)
- Service/model wiring: [app singleton initialization](finsentnet_pro/backend/api/main.py#L144)

## 2. Core model forward path
- Top-level model: [class FinSentNetCore(nn.Module)](finsentnet_pro/backend/models/finsentnet_core.py#L35)
- Core init: [def __init__](finsentnet_pro/backend/models/finsentnet_core.py#L48)
- Core forward: [def forward](finsentnet_pro/backend/models/finsentnet_core.py#L85)
- Text encoder callsite: [sentiment_vec, text_attn = self.text_branch(...)](finsentnet_pro/backend/models/finsentnet_core.py#L109)
- Price encoder callsite: [price_ctx, price_seq = self.price_branch(...)](finsentnet_pro/backend/models/finsentnet_core.py#L112)
- Fusion callsite: [fused_vec, cross_attn = self.fusion(...)](finsentnet_pro/backend/models/finsentnet_core.py#L115)
- Output heads callsite: [head_out = self.dual_head(...)](finsentnet_pro/backend/models/finsentnet_core.py#L118)

## 3. Training path
- Trainer entry: [def train](finsentnet_pro/backend/training/trainer.py#L108)
- Dataset construction: [StockTradingDataset(...)](finsentnet_pro/backend/training/trainer.py#L163)
- Temporal split: [temporal_train_val_split](finsentnet_pro/backend/training/trainer.py#L176)
- Main loop: [def _training_loop](finsentnet_pro/backend/training/trainer.py#L292)
- Validation loop: [def _validate](finsentnet_pro/backend/training/trainer.py#L421)
- Checkpoint save: [def _save_checkpoint](finsentnet_pro/backend/training/trainer.py#L482)
- Checkpoint load: [def _load_checkpoint](finsentnet_pro/backend/training/trainer.py#L517)

## 4. Live inference path
- Predictor entry: [def predict](finsentnet_pro/backend/training/predictor.py#L95)
- Model availability guard: [load_trained_model](finsentnet_pro/backend/training/predictor.py#L137)
- Feature prep and normalization: [price_window extraction](finsentnet_pro/backend/training/predictor.py#L170)
- Inference call: [raw_output = self.model(text_tokens, price_tensor)](finsentnet_pro/backend/training/predictor.py#L197)
- Signal generation: [generate_signal](finsentnet_pro/backend/training/predictor.py#L211)

## 5. Signal post-processing
- Signal API: [class SignalGenerator](finsentnet_pro/backend/models/signal_generator.py#L44)
- Main signal function: [def generate_signal](finsentnet_pro/backend/models/signal_generator.py#L58)
- Direction logic: [def _classify_direction](finsentnet_pro/backend/models/signal_generator.py#L144)
- Regime logic: [def _detect_regime](finsentnet_pro/backend/models/signal_generator.py#L150)
- Technical scoring: [def _technical_score](finsentnet_pro/backend/models/signal_generator.py#L161)

## 6. Frontend consumer path
- Analysis payload builder: [getAnalysisPayload](finsentnet_pro/frontend/js/main.js#L901)
- Signature / cache key: [getAnalysisSignature](finsentnet_pro/frontend/js/main.js#L913)
- Live merge: [mergePredictTunnelIntoState](finsentnet_pro/frontend/js/main.js#L1323)
- Live chart updates: [applyLiveTickToChart](finsentnet_pro/frontend/js/main.js#L1363)
- Dashboard render: [renderDashboard](finsentnet_pro/frontend/js/main.js#L1705)

## 7. Legacy research model (parallel codepath)
- Full research model: [class FinSentNet(nn.Module)](finsent/models/finsent_net.py#L45)
- Research forward: [def forward](finsent/models/finsent_net.py#L153)
- Builder from config: [def from_config](finsent/models/finsent_net.py#L250)

## 8. Complete function index
- Full map (all discovered classes/functions): [MODEL_FUNCTION_INDEX.md](MODEL_FUNCTION_INDEX.md)
