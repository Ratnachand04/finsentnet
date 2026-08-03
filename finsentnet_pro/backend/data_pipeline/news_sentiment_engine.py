"""
FINSENT NET PRO — News Sentiment Engine
FinBERT-based sentiment scoring for financial news headlines.
Falls back to rule-based scoring when transformer is unavailable.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class NewsSentimentEngine:
    """
    Multi-method sentiment analysis for financial text.
    Primary: FinBERT transformer (if available)
    Fallback: Lexicon-based scoring with financial word lists
    """

    # Financial sentiment lexicon (subset — production would use full Loughran-McDonald)
    POSITIVE_WORDS = {
        "beat", "beats", "exceeded", "exceeds", "surpass", "outperform",
        "upgrade", "upgraded", "buy", "bullish", "growth", "profit",
        "revenue", "record", "high", "surge", "rally", "gain", "gains",
        "strong", "positive", "optimistic", "recovery", "improve",
        "expand", "expansion", "dividend", "buyback", "acquisition",
        "breakthrough", "innovation", "momentum", "accelerate",
    }

    NEGATIVE_WORDS = {
        "miss", "missed", "decline", "declined", "fall", "fell", "drop",
        "downgrade", "downgraded", "sell", "bearish", "loss", "losses",
        "weak", "negative", "pessimistic", "recession", "layoff",
        "restructuring", "warning", "risk", "concern", "debt",
        "default", "bankruptcy", "fraud", "investigation", "lawsuit",
        "crash", "plunge", "tumble", "slump", "deteriorate",
    }

    INTENSIFIERS = {
        "significantly", "dramatically", "sharply", "record",
        "unprecedented", "massive", "huge", "enormous",
    }

    def __init__(self, use_finbert: bool = False):
        self.use_finbert = use_finbert
        self.finbert_pipeline = None

        if use_finbert:
            try:
                from transformers import pipeline
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    top_k=3,
                )
                print("[INFO] FinBERT loaded successfully")
            except Exception as e:
                print(f"[WARNING] FinBERT unavailable, using lexicon fallback: {e}")
                self.use_finbert = False

    def score_text(self, text: str) -> Dict:
        """
        Score a single text for sentiment.
        Returns: {score: float (-1 to 1), label: str, confidence: float}
        """
        if self.use_finbert and self.finbert_pipeline:
            return self._finbert_score(text)
        return self._lexicon_score(text)

    def score_batch(self, texts: List[str]) -> List[Dict]:
        """Score multiple texts."""
        return [self.score_text(t) for t in texts]

    def aggregate_sentiment(self, scores: List[Dict]) -> Dict:
        """Aggregate multiple sentiment scores into a single composite."""
        if not scores:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0, "count": 0}

        values = [s["score"] for s in scores]
        confidences = [s["confidence"] for s in scores]

        avg_score = np.mean(values)
        avg_conf = np.mean(confidences)

        if avg_score > 0.15:
            label = "positive"
        elif avg_score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return {
            "score": round(float(avg_score), 4),
            "label": label,
            "confidence": round(float(avg_conf), 4),
            "count": len(scores),
            "positive_pct": round(sum(1 for s in scores if s["label"] == "positive") / len(scores) * 100, 1),
            "negative_pct": round(sum(1 for s in scores if s["label"] == "negative") / len(scores) * 100, 1),
        }

    def _finbert_score(self, text: str) -> Dict:
        """Score using FinBERT transformer."""
        results = self.finbert_pipeline(text[:512])
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]

        score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        best = max(results, key=lambda x: x["score"])

        label = best["label"].lower()
        confidence = best["score"]
        sentiment_val = score_map.get(label, 0.0) * confidence

        return {"score": round(sentiment_val, 4), "label": label, "confidence": round(confidence, 4)}

    def _lexicon_score(self, text: str) -> Dict:
        """Rule-based sentiment scoring using financial lexicon."""
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        intensifier_count = sum(1 for w in words if w in self.INTENSIFIERS)

        total = pos_count + neg_count
        if total == 0:
            return {"score": 0.0, "label": "neutral", "confidence": 0.3}

        raw_score = (pos_count - neg_count) / total
        # Intensifiers amplify magnitude
        boost = 1.0 + 0.2 * min(intensifier_count, 3)
        score = np.clip(raw_score * boost, -1.0, 1.0)

        confidence = min(0.4 + total * 0.1, 0.85)

        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return {"score": round(float(score), 4), "label": label, "confidence": round(float(confidence), 4)}

    def generate_demo_sentiment(self, ticker: str) -> Dict:
        """Generate realistic demo sentiment data for a ticker."""
        np.random.seed(abs(hash(ticker)) % 2**31)
        score = np.random.uniform(-0.5, 0.8)
        conf = np.random.uniform(0.55, 0.92)

        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        headlines = [
            f"{ticker} reports strong quarterly earnings, beating analyst expectations",
            f"Analysts upgrade {ticker} citing robust growth pipeline",
            f"{ticker} announces strategic partnership, shares rise in pre-market",
            f"Market outlook for {ticker} remains cautiously optimistic",
            f"{ticker} expands into new markets, revenue acceleration expected",
        ]

        return {
            "score": round(float(score), 4),
            "label": label,
            "confidence": round(float(conf), 4),
            "normalized_score": round(float((score + 1) / 2 * 100), 1),
            "headline_count": np.random.randint(5, 25),
            "sample_headlines": headlines[:3],
        }
