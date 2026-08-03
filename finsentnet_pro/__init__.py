"""
FINSENT NET PRO — AI-Powered Quantitative Trading Intelligence

Architecture:
    TextBranch (FinBERT → TextCNN → BiLSTM → SelfAttention)
    + PriceBranch (MultiScaleConv1D → DilatedCNN → LSTM)
    → CrossModalAttentionFusion (Q=sentiment, KV=price)
    → DualHeadOutput (Direction + Return)
"""

__version__ = "1.0.0"
