"""
Airflow DAG for FinSentNet Data Ingestion.
=========================================

Handles nightly ingestion of raw data:
1. Fetch OHLCV price data
2. Fetch RSS/API news data
3. Deduplicate via SimHash
4. Trigger subsequent processing

Schedule: Runs at midnight (00:00) UTC daily.
"""

import os
from datetime import datetime, timedelta
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.models import Variable
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False
    # Define stubs for linting if Airflow is not installed locally
    DAG = object
    PythonOperator = object

from finsent.data.price_loader import PriceDataLoader
from finsent.data.news_loader import NewsDataLoader
from finsent.data.text_cleaning import SimHashDeduplicator

default_args = {
    'owner': 'finsent',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

if HAS_AIRFLOW:
    dag = DAG(
        'finsent_daily_ingestion',
        default_args=default_args,
        description='Daily data ingestion for FinSentNet',
        schedule_interval='0 0 * * *',  # Midnight UTC
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['finsent', 'ingestion'],
    )

def fetch_daily_prices(**context):
    """Fetch previous day's OHLCV data."""
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    print(f"Fetching prices for date: {date_str}")
    
    loader = PriceDataLoader(
        cache_dir="/opt/airflow/data/raw",
        processed_dir="/opt/airflow/data/processed"
    )
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "QQQ"]
    
    # Ideally fetch just the delta, or ensure yfinance doesn't over-download
    for ticker in tickers:
        try:
            loader.fetch_yahoo(ticker, start="2010-01-01", end=date_str, force_refresh=True)
            print(f"Successfully fetched prices for {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

def fetch_and_dedup_news(**context):
    """Fetch news from APIs/RSS and deduplicate."""
    execution_date = context['execution_date']
    print(f"Fetching news for {execution_date}")
    
    # In a real setup, fetch from NewsAPI, Bloomberg, Reuters scrapers, etc.
    # Here we mock the ingestion process and apply the SimHash deduplicator.
    raw_news = [
        "Apple reports record earnings despite supply chain issues.",
        "Apple announces record quarterly results amid chip shortage.",  # Near duplicate
        "Microsoft acquires new cloud security firm for $1.2B.",
    ]
    
    deduper = SimHashDeduplicator(threshold=3)
    
    # Normally we load existing fingerprints from prior runs using a DB
    
    unique_news, dupe_count = deduper.deduplicate_batch(raw_news)
    print(f"Found {len(unique_news)} unique articles, removed {dupe_count} duplicates.")
    
    # Save to raw storage path...
    # (Here we would write to Postgres / S3)

if HAS_AIRFLOW:
    price_task = PythonOperator(
        task_id='fetch_daily_prices',
        python_callable=fetch_daily_prices,
        provide_context=True,
        dag=dag,
    )

    news_task = PythonOperator(
        task_id='fetch_and_dedup_news',
        python_callable=fetch_and_dedup_news,
        provide_context=True,
        dag=dag,
    )

    # Simple parallel execution
    [price_task, news_task]
