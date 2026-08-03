"""
Text Cleaning Pipeline — Step 4 of Data Pipeline.
===================================================

Implements the full text cleaning pipeline for financial news:
    1. HTML tag removal
    2. Sentence tokenization
    3. Financial entity normalization ($AAPL, Apple Inc. → <TICKER>)
    4. FinBERT tokenization via the text_branch_v2 tokenizer

This module bridges raw news text from ingestion to model input format.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Financial Entity Patterns
# ═══════════════════════════════════════════════════════════════════════

# Common ticker patterns: $AAPL, $MSFT, etc.
TICKER_CASH_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

# Known company → ticker mapping (extensible)
COMPANY_TO_TICKER = {
    "apple inc": "AAPL", "apple": "AAPL",
    "microsoft": "MSFT", "microsoft corp": "MSFT",
    "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "amazon.com": "AMZN",
    "meta platforms": "META", "facebook": "META",
    "nvidia": "NVDA", "nvidia corp": "NVDA",
    "tesla": "TSLA", "tesla inc": "TSLA",
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpmorgan chase": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "berkshire hathaway": "BRK",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "procter & gamble": "PG", "procter and gamble": "PG",
    "walt disney": "DIS", "disney": "DIS",
    "coca-cola": "KO", "coca cola": "KO",
    "visa inc": "V", "mastercard": "MA",
    "federal reserve": "FED", "the fed": "FED",
    "securities and exchange commission": "SEC",
    "s&p 500": "SPX", "s&p500": "SPX", "sp500": "SPX",
    "nasdaq": "NDX", "dow jones": "DJI",
}

# Financial abbreviation patterns to preserve
FINANCIAL_ABBREVIATIONS = {
    "ebitda", "eps", "p/e", "pe ratio", "roe", "roa",
    "gdp", "cpi", "ppi", "fomc", "ipo", "etf", "reit",
    "yoy", "qoq", "mom", "ytd", "mtd",
    "bps", "basis points", "fed funds",
    "short squeeze", "dead cat bounce", "fed pivot",
    "quantitative easing", "quantitative tightening",
    "yield curve", "inverted yield curve",
    "market cap", "book value", "free cash flow",
}

# Monetary value patterns
MONEY_PATTERN = re.compile(
    r'\$\s*(\d+(?:\.\d+)?)\s*(million|billion|trillion|mn|bn|tn|m|b|t)?',
    re.IGNORECASE
)

# Percentage patterns
PERCENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(%|percent|pct|basis\s*points|bps)')


class FinancialTextCleaner:
    """Comprehensive text cleaning pipeline for financial news.
    
    Pipeline:
        Raw HTML text
            → Strip HTML tags
            → Unicode normalization
            → Sentence splitting
            → Entity normalization (companies → <TICKER>)
            → Number normalization (monetary, percentages)
            → Whitespace cleanup
            → Ready for FinBERT tokenization
    
    Design Principles:
        1. Preserve financial meaning: "$AAPL" and "Apple Inc." both → <TICKER_AAPL>
        2. Normalize numbers: "$1.2 billion" → <MONEY>, "3.5%" → <PERCENT>
        3. Keep sentiment words intact: "beat", "miss", "warn" are critical signals
        4. Handle HTML artifacts from web scraping
    """
    
    def __init__(
        self,
        normalize_entities: bool = True,
        normalize_numbers: bool = True,
        max_sentence_length: int = 512,
        custom_entity_map: Optional[Dict[str, str]] = None,
    ):
        self.normalize_entities = normalize_entities
        self.normalize_numbers = normalize_numbers
        self.max_sentence_length = max_sentence_length
        
        # Merge custom entity mappings
        self.entity_map = dict(COMPANY_TO_TICKER)
        if custom_entity_map:
            self.entity_map.update(custom_entity_map)
        
        # Compile company name pattern for efficient matching
        # Sort by length (longest first) to match "Bank of America" before "Bank"
        sorted_names = sorted(self.entity_map.keys(), key=len, reverse=True)
        escaped = [re.escape(name) for name in sorted_names]
        self.company_pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE
        )
    
    def clean(self, text: str) -> str:
        """Full cleaning pipeline for a single text.
        
        Args:
            text: Raw text (may contain HTML, special chars, etc.)
        
        Returns:
            Cleaned text ready for tokenization
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: HTML tag removal
        text = self._remove_html(text)
        
        # Step 2: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 3: URL removal
        text = re.sub(r'https?://\S+', ' <URL> ', text)
        text = re.sub(r'www\.\S+', ' <URL> ', text)
        
        # Step 4: Email removal
        text = re.sub(r'\S+@\S+\.\S+', ' <EMAIL> ', text)
        
        # Step 5: Entity normalization
        if self.normalize_entities:
            text = self._normalize_entities(text)
        
        # Step 6: Number normalization
        if self.normalize_numbers:
            text = self._normalize_numbers(text)
        
        # Step 7: Whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 8: Truncate if needed
        if len(text) > self.max_sentence_length * 5:  # rough char limit
            text = text[:self.max_sentence_length * 5]
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(text) for text in texts]
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags, decode entities."""
        # Remove script/style blocks
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&nbsp;', ' ')
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # NFKD decomposition + filter non-ASCII selectively
        text = unicodedata.normalize('NFKD', text)
        # Keep common financial symbols
        text = re.sub(r'[^\x00-\x7F€£¥°±×÷]', ' ', text)
        return text
    
    def _normalize_entities(self, text: str) -> str:
        """Replace company names and ticker symbols with normalized tokens."""
        # Replace $TICKER patterns
        text = TICKER_CASH_PATTERN.sub(r'<TICKER_\1>', text)
        
        # Replace company names with ticker tokens
        def replace_company(match):
            name = match.group(1).lower()
            ticker = self.entity_map.get(name, name.upper())
            return f'<TICKER_{ticker}>'
        
        text = self.company_pattern.sub(replace_company, text)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Normalize monetary values and percentages."""
        # Replace monetary values: $1.2 billion → <MONEY>
        text = MONEY_PATTERN.sub(' <MONEY> ', text)
        
        # Replace percentages: 3.5% → <PERCENT>
        text = PERCENT_PATTERN.sub(' <PERCENT> ', text)
        
        # Replace remaining standalone large numbers
        text = re.sub(r'\b\d{6,}\b', ' <BIGNUM> ', text)
        
        return text
    
    @staticmethod
    def sentence_tokenize(text: str) -> List[str]:
        """Split text into sentences.
        
        Simple rule-based approach that handles:
        - Standard period/question/exclamation endings
        - Abbreviations (Mr., Mrs., Inc., Corp., etc.)
        - Decimal numbers (3.14 is not a sentence boundary)
        """
        # Protect abbreviations
        abbrevs = r'(?:Mr|Mrs|Ms|Dr|Prof|Inc|Corp|Ltd|Co|Jr|Sr|vs|etc|vol|no)\.'
        text = re.sub(f'({abbrevs})', lambda m: m.group().replace('.', '<DOT>'), text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter empty/too-short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences


class SimHashDeduplicator:
    """Content deduplication using SimHash fingerprinting.
    
    SimHash (Charikar, 2002) generates a fixed-size fingerprint for each
    document. Documents with similar content produce similar hashes.
    Two documents are considered duplicates if their Hamming distance
    is below a threshold.
    
    This is used in Step 1 (Ingestion) to prevent duplicate articles
    from inflating the dataset and biasing the model.
    
    Complexity: O(n × d) for n documents with d features.
    Space: O(n × 64 bits) for storing fingerprints.
    """
    
    def __init__(self, hash_bits: int = 64, threshold: int = 3):
        self.hash_bits = hash_bits
        self.threshold = threshold
        self.fingerprints: List[int] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization for hashing."""
        return text.lower().split()
    
    def _hash_token(self, token: str) -> int:
        """FNV-1a hash for a single token."""
        h = 0xcbf29ce484222325  # FNV offset basis
        for byte in token.encode('utf-8'):
            h ^= byte
            h = (h * 0x100000001b3) & ((1 << self.hash_bits) - 1)
        return h
    
    def compute_fingerprint(self, text: str) -> int:
        """Compute SimHash fingerprint for a document.
        
        Algorithm:
        1. Tokenize document
        2. Hash each token to hash_bits-length hash
        3. For each bit position, sum +1 if bit=1, -1 if bit=0
        4. Final fingerprint: 1 where sum > 0, 0 otherwise
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0
        
        # Accumulator for each bit position
        v = [0] * self.hash_bits
        
        for token in tokens:
            h = self._hash_token(token)
            for i in range(self.hash_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Build fingerprint
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def hamming_distance(self, fp1: int, fp2: int) -> int:
        """Count differing bits between two fingerprints."""
        xor = fp1 ^ fp2
        return bin(xor).count('1')
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate of any existing document.
        
        Returns True if Hamming distance to any stored fingerprint
        is <= threshold.
        """
        fp = self.compute_fingerprint(text)
        
        for existing_fp in self.fingerprints:
            if self.hamming_distance(fp, existing_fp) <= self.threshold:
                return True
        
        # Not a duplicate — store the fingerprint
        self.fingerprints.append(fp)
        return False
    
    def deduplicate_batch(self, texts: List[str]) -> Tuple[List[str], int]:
        """Deduplicate a batch of texts.
        
        Returns:
            (unique_texts, n_duplicates_removed)
        """
        unique = []
        n_dupes = 0
        
        for text in texts:
            if not self.is_duplicate(text):
                unique.append(text)
            else:
                n_dupes += 1
        
        return unique, n_dupes
    
    def reset(self):
        """Clear stored fingerprints."""
        self.fingerprints = []
