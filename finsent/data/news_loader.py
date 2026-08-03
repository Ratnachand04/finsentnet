"""
News / text data loader and preprocessor.
==========================================

Handles:
  - News article loading from CSV/JSON
  - Text cleaning and tokenization (from scratch — no HuggingFace tokenizers)
  - Vocabulary building with frequency thresholds
  - Temporal alignment with price data (prevent look-ahead)
  - Sentiment aggregation over configurable time windows
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict, Tuple


class Vocabulary:
    """Word-level vocabulary built from corpus.
    
    Special tokens:
        <PAD> = 0   — padding
        <UNK> = 1   — unknown / out-of-vocabulary
        <BOS> = 2   — beginning of sequence
        <EOS> = 3   — end of sequence
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    
    def __init__(self, max_size: int = 50000, min_freq: int = 5):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self._built = False
    
    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from list of tokenized text strings.
        
        Steps:
        1. Count word frequencies across all texts
        2. Filter by min_freq
        3. Keep top max_size words
        4. Assign integer indices
        """
        # Count frequencies
        for text in texts:
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)
        
        # Filter and sort
        filtered = {
            w: c for w, c in self.word_freq.items() if c >= self.min_freq
        }
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])
        
        # Build mappings (special tokens first)
        self.word2idx = {}
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.word2idx[token] = i
        
        offset = len(self.SPECIAL_TOKENS)
        for i, (word, _) in enumerate(sorted_words[:self.max_size - offset]):
            self.word2idx[word] = i + offset
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._built = True
        
        print(f"[Vocabulary] Built: {len(self.word2idx)} words "
              f"(from {len(self.word_freq)} unique, min_freq={self.min_freq})")
        return self
    
    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """Convert text to integer token indices.
        
        Adds BOS/EOS tokens, pads/truncates to max_length.
        """
        assert self._built, "Vocabulary not built. Call .build() first."
        
        tokens = self._tokenize(text)
        
        # Add BOS/EOS
        indices = [self.word2idx[self.BOS_TOKEN]]
        for token in tokens[:max_length - 2]:  # leave room for BOS + EOS
            indices.append(self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]))
        indices.append(self.word2idx[self.EOS_TOKEN])
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.word2idx[self.PAD_TOKEN])
        
        return np.array(indices, dtype=np.int64)
    
    def decode(self, indices: np.ndarray) -> str:
        """Convert integer indices back to text."""
        tokens = [self.idx2word.get(int(idx), self.UNK_TOKEN) for idx in indices]
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return " ".join(tokens)
    
    @property
    def size(self) -> int:
        return len(self.word2idx)
    
    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    def save(self, path: str) -> None:
        """Save vocabulary to JSON."""
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f)
    
    def load(self, path: str) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.min_freq = data["min_freq"]
        self._built = True
        return self
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer.
        
        For production, replace with BPE or SentencePiece.
        This suffices for financial news where vocabulary is relatively stable.
        """
        text = text.lower().strip()
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove special chars but keep financial symbols
        text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\%\$]", " ", text)
        # Separate numbers from text
        text = re.sub(r"(\d+)", r" \1 ", text)
        # Collapse whitespace
        tokens = text.split()
        return tokens


class NewsDataLoader:
    """Load and process news/text data for FinSentNet.
    
    Supports:
    - CSV format: columns = [date, headline, body, ticker, source]
    - JSON format: list of news objects
    - Synthetic news generation for testing
    
    CRITICAL: Temporal alignment ensures news at time t only uses
    information available BEFORE time t (configurable lag).
    """
    
    def __init__(
        self,
        max_seq_length: int = 512,
        max_news_per_day: int = 50,
        news_lookback_hours: int = 72,
        max_news_lag_hours: int = 24,
    ):
        self.max_seq_length = max_seq_length
        self.max_news_per_day = max_news_per_day
        self.news_lookback_hours = news_lookback_hours
        self.max_news_lag_hours = max_news_lag_hours
        self.vocab: Optional[Vocabulary] = None
    
    def load_csv(
        self,
        filepath: str,
        date_col: str = "date",
        text_col: str = "headline",
        ticker_col: Optional[str] = "ticker",
    ) -> pd.DataFrame:
        """Load news from CSV file."""
        df = pd.read_csv(filepath, parse_dates=[date_col])
        df = df.rename(columns={date_col: "datetime", text_col: "text"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        # Clean text
        df["text"] = df["text"].fillna("").astype(str).apply(self._clean_text)
        df = df[df["text"].str.len() > 10]  # drop too-short articles
        
        print(f"[NewsLoader] Loaded {len(df)} articles from {filepath}")
        return df
    
    def build_vocabulary(
        self,
        texts: List[str],
        max_size: int = 50000,
        min_freq: int = 5,
    ) -> Vocabulary:
        """Build vocabulary from training texts only (no val/test leakage)."""
        self.vocab = Vocabulary(max_size=max_size, min_freq=min_freq)
        self.vocab.build(texts)
        return self.vocab
    
    def align_news_to_prices(
        self,
        news_df: pd.DataFrame,
        price_dates: pd.DatetimeIndex,
        ticker: Optional[str] = None,
    ) -> Dict[pd.Timestamp, List[str]]:
        """Align news to price dates with temporal safety.
        
        For each price date t:
        - Collect news from [t - news_lookback_hours, t - max_news_lag_hours]
        - This ensures news is available BEFORE the price we're predicting
        
        Returns:
            dict mapping price_date → list of news text strings
        """
        if ticker and "ticker" in news_df.columns:
            news_df = news_df[news_df["ticker"] == ticker]
        
        aligned = {}
        lookback = pd.Timedelta(hours=self.news_lookback_hours)
        lag = pd.Timedelta(hours=self.max_news_lag_hours)
        
        for date in price_dates:
            # News window: [date - lookback, date - lag]
            # This prevents look-ahead: minimum `lag` hours before price observation
            window_start = date - lookback
            window_end = date - lag
            
            mask = (news_df["datetime"] >= window_start) & (news_df["datetime"] <= window_end)
            day_news = news_df.loc[mask, "text"].tolist()
            
            # Limit to most recent N articles
            if len(day_news) > self.max_news_per_day:
                day_news = day_news[-self.max_news_per_day:]
            
            aligned[date] = day_news
        
        coverage = sum(1 for v in aligned.values() if len(v) > 0) / len(aligned)
        print(f"[NewsLoader] Aligned news: {coverage:.1%} of price dates have news coverage")
        
        return aligned
    
    def encode_texts(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Encode list of texts into padded integer arrays.
        
        If multiple texts for same date, concatenate with separator.
        
        Returns: shape (max_seq_length,) integer array
        """
        assert self.vocab is not None, "Vocabulary not built"
        
        if not texts:
            # Return all-PAD for dates with no news
            return np.zeros(self.max_seq_length, dtype=np.int64)
        
        # Concatenate texts with period separator
        combined = " . ".join(texts)
        return self.vocab.encode(combined, max_length=self.max_seq_length)
    
    def generate_synthetic_news(
        self,
        price_df: pd.DataFrame,
        n_articles_per_day: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic news data for testing pipelines.
        
        Creates simple sentiment-correlated news based on price movements.
        NOT for actual model training — only for pipeline validation.
        """
        np.random.seed(seed)
        
        positive_templates = [
            "{ticker} reports strong earnings beating analyst expectations",
            "{ticker} shares rally on positive revenue growth outlook",
            "Analysts upgrade {ticker} citing strong market fundamentals",
            "{ticker} announces expansion plans driving investor optimism",
            "Market momentum pushes {ticker} to new highs",
        ]
        negative_templates = [
            "{ticker} misses earnings estimates sending shares lower",
            "Concerns grow over {ticker} declining market share",
            "{ticker} faces regulatory headwinds as investigation widens",
            "Analysts downgrade {ticker} on weak forward guidance",
            "{ticker} revenue falls short amid challenging macroeconomic conditions",
        ]
        neutral_templates = [
            "{ticker} reports quarterly results in line with expectations",
            "{ticker} maintains steady dividend amid market uncertainty",
            "Trading volume in {ticker} remains within normal range",
            "{ticker} executives present at annual industry conference",
            "{ticker} files routine quarterly report with regulators",
        ]
        
        records = []
        ticker = "SYN"
        
        returns = price_df["Close"].pct_change()
        
        for date, ret in returns.items():
            if np.isnan(ret):
                continue
            
            # Select templates based on return direction
            if ret > 0.005:
                templates = positive_templates
            elif ret < -0.005:
                templates = negative_templates
            else:
                templates = neutral_templates
            
            n_articles = np.random.randint(1, n_articles_per_day + 1)
            for _ in range(n_articles):
                template = np.random.choice(templates)
                text = template.format(ticker=ticker)
                
                # Add some noise: small chance of contrary sentiment
                if np.random.random() < 0.1:
                    alt_templates = positive_templates if ret < 0 else negative_templates
                    text = np.random.choice(alt_templates).format(ticker=ticker)
                
                # News timestamp: random time during prior business day
                hours_offset = np.random.uniform(self.max_news_lag_hours, self.news_lookback_hours)
                news_time = date - pd.Timedelta(hours=hours_offset)
                
                records.append({
                    "datetime": news_time,
                    "text": text,
                    "ticker": ticker,
                })
        
        df = pd.DataFrame(records)
        df = df.sort_values("datetime").reset_index(drop=True)
        print(f"[NewsLoader] Generated {len(df)} synthetic articles")
        return df
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean raw news text."""
        text = re.sub(r"<[^>]+>", "", text)         # Remove HTML
        text = re.sub(r"https?://\S+", "", text)     # Remove URLs
        text = re.sub(r"\s+", " ", text)             # Collapse whitespace
        return text.strip()
