"""
FINSENT NET PRO — Signal Generator
Converts FINSENT model outputs into actionable trade signals
with Kelly Criterion position sizing and risk-adjusted allocation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class SignalDirection(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class TradeSignal:
    ticker: str
    market: str
    direction: SignalDirection
    confidence: float
    predicted_return: float
    predicted_downside: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    kelly_fraction: float
    recommended_quantity: int
    capital_required: float
    time_horizon: str
    reasoning: List[str]
    regime: str
    sentiment_score: float
    technical_score: float


class SignalGenerator:
    """
    Converts raw model outputs into institutional-grade trade signals.
    Uses Kelly Criterion for optimal position sizing.
    """

    DIRECTION_THRESHOLDS = {
        SignalDirection.STRONG_BUY:  (0.70, float("inf")),
        SignalDirection.BUY:         (0.55, 0.70),
        SignalDirection.HOLD:        (0.40, 0.55),
        SignalDirection.SELL:        (0.25, 0.40),
        SignalDirection.STRONG_SELL: (0.0,  0.25),
    }

    def generate_signal(
        self,
        ticker: str,
        market: str,
        model_output: dict,
        price_data: pd.DataFrame,
        total_capital: float,
        risk_tolerance: float = 0.5,
    ) -> TradeSignal:
        """Generate a complete trade signal with position sizing."""
        probs = model_output["direction_probs"]
        if hasattr(probs, "detach"):
            probs = probs.detach().cpu().numpy()
        if probs.ndim == 2:
            probs = probs[0]

        p_up, p_neutral, p_down = float(probs[0]), float(probs[1]), float(probs[2])

        pred_return_raw = model_output["return_pred"]
        if hasattr(pred_return_raw, "detach"):
            pred_return_raw = pred_return_raw.detach().cpu().numpy()
        pred_return = float(pred_return_raw.flatten()[0])

        current_price = float(price_data["Close"].iloc[-1])
        atr = float(price_data["ATR_14"].iloc[-1]) if "ATR_14" in price_data.columns else current_price * 0.02

        direction = self._classify_direction(p_up)

        atr_mult_target = 2.5 + risk_tolerance * 1.5
        atr_mult_stop = 1.5 + (1 - risk_tolerance) * 1.0

        if direction in (SignalDirection.STRONG_BUY, SignalDirection.BUY):
            target_price = current_price * (1 + abs(pred_return))
            stop_loss = current_price - (atr * atr_mult_stop)
            reward = target_price - current_price
            risk = current_price - stop_loss
            p_win = p_up
        elif direction in (SignalDirection.STRONG_SELL, SignalDirection.SELL):
            target_price = current_price * (1 - abs(pred_return))
            stop_loss = current_price + (atr * atr_mult_stop)
            reward = current_price - target_price
            risk = stop_loss - current_price
            p_win = p_down
        else:
            target_price = current_price * 1.02
            stop_loss = current_price * 0.98
            reward = target_price - current_price
            risk = current_price - stop_loss
            p_win = max(p_up, p_down)

        rr_ratio = reward / max(risk, 0.001)

        # Kelly: f* = (b*p - q) / b   capped at 25%
        q = 1 - p_win
        kelly = (rr_ratio * p_win - q) / max(rr_ratio, 0.001)
        kelly = max(0, min(kelly, 0.25)) * max(risk_tolerance, 0.1)

        capital_to_deploy = total_capital * kelly
        quantity = max(1, int(capital_to_deploy / current_price)) if current_price > 0 else 0
        actual_capital = quantity * current_price

        reasoning = self._generate_reasoning(
            ticker, direction, p_up, p_down, pred_return, price_data, p_win * 100
        )

        return TradeSignal(
            ticker=ticker,
            market=market,
            direction=direction,
            confidence=round(p_win * 100, 1),
            predicted_return=round(pred_return * 100, 2),
            predicted_downside=round(-abs(pred_return) * 0.5 * 100, 2),
            entry_price=round(current_price, 2),
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            kelly_fraction=round(kelly, 4),
            recommended_quantity=quantity,
            capital_required=round(actual_capital, 2),
            time_horizon=self._estimate_horizon(pred_return, atr, current_price),
            reasoning=reasoning,
            regime=self._detect_regime(price_data),
            sentiment_score=round(float(p_up * 100), 1),
            technical_score=round(self._technical_score(price_data), 1),
        )

    def _classify_direction(self, p_up: float) -> SignalDirection:
        for direction, (low, high) in self.DIRECTION_THRESHOLDS.items():
            if low <= p_up < high:
                return direction
        return SignalDirection.HOLD

    def _detect_regime(self, price_data: pd.DataFrame) -> str:
        if "SMA_50" in price_data.columns and "SMA_200" in price_data.columns:
            latest = price_data.iloc[-1]
            if latest["Close"] > latest["SMA_50"] > latest["SMA_200"]:
                vol = latest.get("Volatility_20", 0.2)
                return "BULL — Low Volatility" if vol < 0.20 else "BULL — Normal"
            elif latest["Close"] < latest["SMA_50"] < latest["SMA_200"]:
                return "BEAR"
            return "TRANSITIONAL"
        return "UNKNOWN"

    def _technical_score(self, price_data: pd.DataFrame) -> float:
        score = 50.0
        latest = price_data.iloc[-1]
        if "RSI_14" in price_data.columns:
            rsi = latest["RSI_14"]
            if rsi < 30:
                score += 20
            elif 30 <= rsi <= 70:
                score += 10
        if "MACD" in price_data.columns and "MACD_Signal" in price_data.columns:
            if latest["MACD"] > latest["MACD_Signal"]:
                score += 15
        if "GoldenCross" in price_data.columns and latest["GoldenCross"] == 1:
            score += 10
        if "BB_Position" in price_data.columns:
            if latest["BB_Position"] < 0.2:
                score += 15
        return min(100, max(0, score))

    def _estimate_horizon(self, pred_return: float, atr: float, price: float) -> str:
        daily_vol = atr / max(price, 1)
        if abs(pred_return) < daily_vol * 2:
            return "1-3 days"
        elif abs(pred_return) < daily_vol * 10:
            return "1-2 weeks"
        elif abs(pred_return) < daily_vol * 25:
            return "3-6 weeks"
        return "2-3 months"

    def _generate_reasoning(self, ticker, direction, p_up, p_down,
                            pred_return, price_data, confidence) -> List[str]:
        reasons = []
        latest = price_data.iloc[-1]
        reasons.append(f"FINSENT confidence: {confidence:.1f}% probability of {direction.value}")
        reasons.append(f"Predicted price movement: {'+' if pred_return > 0 else ''}{pred_return*100:.2f}%")

        if "RSI_14" in price_data.columns:
            rsi = latest["RSI_14"]
            if rsi < 35:
                reasons.append(f"RSI({rsi:.0f}) — Oversold territory, historically bullish reversal zone")
            elif rsi > 70:
                reasons.append(f"RSI({rsi:.0f}) — Overbought, potential for near-term pullback")
            else:
                reasons.append(f"RSI({rsi:.0f}) — Neutral momentum zone")

        if "MACD" in price_data.columns and "MACD_Signal" in price_data.columns:
            cross = "bullish" if latest["MACD"] > latest["MACD_Signal"] else "bearish"
            reasons.append(f"MACD showing {cross} momentum configuration")

        if "GoldenCross" in price_data.columns:
            cross_type = ("Golden Cross (50 SMA > 200 SMA)" if latest["GoldenCross"]
                          else "Death Cross (50 SMA < 200 SMA)")
            reasons.append(f"Trend structure: {cross_type}")

        if "Volume_Ratio" in price_data.columns and latest["Volume_Ratio"] > 1.5:
            reasons.append(f"Volume {latest['Volume_Ratio']:.1f}x above avg — institutional interest")

        return reasons[:5]
