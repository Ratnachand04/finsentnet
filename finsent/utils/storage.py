"""
Storage management for the data pipeline.
========================================

Step 6 of Data Pipeline:
- PostgreSQL for metadata + labels
- Parquet partitioned by ticker + year for price data
- HuggingFace datasets format for text
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

class PipelineStorage:
    """Handles storage of pipeline artifacts."""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.price_dir = self.base_dir / "prices_parquet"
        self.text_dir = self.base_dir / "hf_datasets"
        self.meta_dir = self.base_dir / "metadata"
        
        for d in [self.price_dir, self.text_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_price_data(self, df: pd.DataFrame, ticker: str):
        """Save price data to Parquet partitioned by year."""
        # Ensure 'Date' index is a column for partitioning
        if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
            df_save = df.reset_index()
            # Rename index column to 'Date' if it dropped its name
            if 'index' in df_save.columns:
                df_save = df_save.rename(columns={'index': 'Date'})
        else:
            df_save = df.copy()
            
        df_save['year'] = df_save['Date'].dt.year
        
        # Save partitioned parquet
        output_path = self.price_dir / ticker
        df_save.to_parquet(
            output_path,
            partition_cols=['year'],
            engine='pyarrow',
            index=False
        )
        print(f"[Storage] Saved partitioned Parquet for {ticker} at {output_path}")

    def save_text_dataset(self, text_records: List[Dict[str, Any]], dataset_name: str):
        """Save text data in HuggingFace Datasets structure."""
        if not HAS_DATASETS:
            print("[Storage] `datasets` library not installed. Can't save HF format.")
            return
            
        df = pd.DataFrame(text_records)
        dataset = datasets.Dataset.from_pandas(df)
        
        output_path = self.text_dir / dataset_name
        dataset.save_to_disk(str(output_path))
        print(f"[Storage] Saved HF Dataset {dataset_name} at {output_path}")

    def save_metadata(self, metadata: Dict, name: str):
        """Save pipeline metadata. In prod: saves to PostgreSQL."""
        # For the standalone pipeline, we simulate PG with complete JSON dumps
        # Real prod would use psycopg2/SQLAlchemy here
        output_path = self.meta_dir / f"{name}_metadata.json"
        
        # Handle datetime serialization
        def default_serialize(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
            
        import numpy as np
        with open(output_path, 'w') as f:
            json.dump(metadata, f, default=default_serialize, indent=2)
            
        print(f"[Storage] Saved metadata: {output_path}")

