#!/usr/bin/env python3
"""
FinSentNet — Main Entry Point
===============================

End-to-end pipeline orchestrator:
  1. Load configuration
  2. Set seeds / device
  3. Run data pipeline (fetch → features → align → split → loaders)
  4. Construct model from config
  5. Train with early stopping on validation Sharpe
  6. Post-training temperature calibration
  7. Run backtest on held-out test set
  8. Generate performance report & visualizations

Usage:
    python main.py                          # full run with defaults
    python main.py --config path/cfg.yaml   # custom config
    python main.py --mode train             # training only
    python main.py --mode backtest          # backtest from checkpoint
    python main.py --ticker AAPL            # single ticker
"""

import os
import sys
import yaml
import argparse
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from finsent.utils.seed import set_seed, get_device
from finsent.utils.logging_utils import setup_logger
from finsent.data.pipeline import DataPipeline
from finsent.models.finsent_net import FinSentNet
from finsent.training.trainer import Trainer
from finsent.backtest.engine import BacktestEngine
from finsent.backtest.metrics import print_metrics
from finsent.utils.visualization import (
    plot_equity_curve,
    plot_returns_distribution,
    plot_attention_heatmap,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FinSentNet — Financial Sentiment Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "train", "backtest", "data_only"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Single ticker override (default: use config list).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint for backtest mode.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for outputs (models, plots, logs).",
    )
    parser.add_argument(
        "--no_synthetic_news", action="store_true",
        help="Disable synthetic news generation (requires real news CSV).",
    )
    parser.add_argument(
        "--news_csv", type=str, default=None,
        help="Path to real news CSV (columns: datetime, text, ticker).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and validate configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    required_sections = ["data", "text_branch", "price_branch", "fusion",
                          "dual_head", "training", "backtest"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: '{section}'")
    
    return config


def setup_output_dirs(output_dir: str) -> dict:
    """Create output directory structure."""
    dirs = {
        "root": output_dir,
        "checkpoints": os.path.join(output_dir, "checkpoints"),
        "plots": os.path.join(output_dir, "plots"),
        "logs": os.path.join(output_dir, "logs"),
        "results": os.path.join(output_dir, "results"),
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs


def run_data_pipeline(config: dict, args, logger) -> tuple:
    """Execute data pipeline."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA PIPELINE")
    logger.info("=" * 70)
    
    pipeline = DataPipeline(args.config)
    
    ticker = args.ticker or config["data"]["tickers"][0]
    use_synthetic = not args.no_synthetic_news
    
    logger.info(f"  Ticker: {ticker}")
    logger.info(f"  Date range: {config['data']['start_date']} → {config['data']['end_date']}")
    logger.info(f"  Synthetic news: {use_synthetic}")
    
    train_loader, val_loader, test_loader, metadata = pipeline.run(
        ticker=ticker,
        use_synthetic_news=use_synthetic,
        news_csv_path=args.news_csv,
    )
    
    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Val samples:   {len(val_loader.dataset)}")
    logger.info(f"  Test samples:  {len(test_loader.dataset)}")
    logger.info(f"  Vocab size:    {metadata['vocab_size']}")
    
    return train_loader, val_loader, test_loader, metadata


def build_model(config: dict, metadata: dict, device: torch.device, logger) -> FinSentNet:
    """Construct and initialize model."""
    logger.info("=" * 70)
    logger.info("PHASE 2: MODEL CONSTRUCTION")
    logger.info("=" * 70)
    
    # Override vocab size with actual vocabulary size
    if metadata.get("vocab_size", 0) > 0:
        config["data"]["vocab_size"] = metadata["vocab_size"]
    
    model = FinSentNet.from_config(config)
    model = model.to(device)
    model.print_architecture()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters:     {total_params:>12,d}")
    logger.info(f"  Trainable parameters: {trainable_params:>12,d}")
    logger.info(f"  Device: {device}")
    
    return model


def run_training(
    model: FinSentNet,
    config: dict,
    train_loader,
    val_loader,
    device: torch.device,
    dirs: dict,
    logger,
) -> dict:
    """Execute training loop."""
    logger.info("=" * 70)
    logger.info("PHASE 3: TRAINING")
    logger.info("=" * 70)
    
    trainer = Trainer(model, config, device)
    
    start = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - start
    
    logger.info(f"  Training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"  Best epoch: {trainer.early_stopping.best_epoch}")
    logger.info(f"  Best val Sharpe: {trainer.early_stopping.best_score:.4f}")
    
    # Post-training calibration
    logger.info("  Running temperature calibration...")
    calibration_results = trainer.post_training_calibrate(val_loader)
    logger.info(f"  ECE before: {calibration_results.get('ece_before', 'N/A')}")
    logger.info(f"  ECE after:  {calibration_results.get('ece_after', 'N/A')}")
    
    # Save final calibrated model
    final_path = os.path.join(dirs["checkpoints"], "finsent_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
        "calibration": calibration_results,
    }, final_path)
    logger.info(f"  Final model saved to {final_path}")
    
    return history


def run_backtest(
    model: FinSentNet,
    config: dict,
    test_loader,
    metadata: dict,
    device: torch.device,
    dirs: dict,
    logger,
):
    """Execute backtest on test set."""
    logger.info("=" * 70)
    logger.info("PHASE 4: BACKTESTING")
    logger.info("=" * 70)
    
    bcfg = config["backtest"]
    
    engine = BacktestEngine(
        initial_capital=bcfg["initial_capital"],
        commission_bps=bcfg["commission_bps"],
        slippage_bps=bcfg["slippage_bps"],
        max_position_pct=bcfg["max_position_pct"],
        kelly_fraction=bcfg["kelly_fraction"],
        risk_free_rate=bcfg.get("risk_free_rate", 0.05),
    )
    
    # Extract test dates and close prices from dataset
    test_dataset = test_loader.dataset
    test_dates = []
    test_close = []
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        # Use the last close price in the window as a reference
        price_window = sample["price"]  # (window_size, n_features)
        # Close is feature index 3 (OHLCV: Open=0, High=1, Low=2, Close=3, Volume=4)
        last_close = price_window[-1, 3].item()
        test_close.append(last_close)
    
    # Generate placeholder dates if not available in dataset
    n_test = len(test_dataset)
    test_dates_pd = pd.date_range(
        start=metadata.get("test_dates", (pd.Timestamp("2023-01-01"),))[0],
        periods=n_test,
        freq="B",  # business days
    )
    
    result = engine.run(
        model=model,
        test_loader=test_loader,
        test_dates=test_dates_pd.tolist(),
        test_close_prices=np.array(test_close) if test_close else np.ones(n_test),
        device=device,
    )
    
    # Display metrics
    logger.info("\n  BACKTEST RESULTS:")
    print_metrics(result.metrics)
    
    # Save results
    results_path = os.path.join(dirs["results"], "backtest_results.npz")
    np.savez(
        results_path,
        equity=result.equity_curve.values,
        returns=result.returns.values,
        drawdown=result.drawdown.values,
        predictions=result.predictions,
        confidences=result.confidences,
        metrics=result.metrics,
    )
    logger.info(f"  Results saved to {results_path}")
    
    # Generate plots
    try:
        plot_equity_curve(
            equity=result.equity_curve,
            drawdown=result.drawdown,
            title=f"FinSentNet Backtest — {metadata.get('ticker', 'Unknown')}",
            save_path=os.path.join(dirs["plots"], "equity_curve.png"),
        )
        
        plot_returns_distribution(
            returns=result.returns,
            title="Strategy Returns Distribution",
            save_path=os.path.join(dirs["plots"], "returns_dist.png"),
        )
        
        logger.info(f"  Plots saved to {dirs['plots']}")
    except Exception as e:
        logger.warning(f"  Plot generation failed: {e}")
    
    return result


def print_summary(metadata: dict, history: dict, result, logger):
    """Print final summary report."""
    logger.info("\n" + "=" * 70)
    logger.info("FINSENT NET — EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Ticker:            {metadata.get('ticker', 'N/A')}")
    logger.info(f"  Train period:      {metadata.get('train_dates', ('N/A', 'N/A'))}")
    logger.info(f"  Test period:       {metadata.get('test_dates', ('N/A', 'N/A'))}")
    
    if history:
        best_epoch_idx = np.argmax(history.get("val_sharpe", [0]))
        logger.info(f"  Best training epoch: {best_epoch_idx + 1}")
        logger.info(f"  Best val Sharpe:     {max(history.get('val_sharpe', [0])):.4f}")
        logger.info(f"  Best val accuracy:   {max(history.get('val_accuracy', [0])):.4f}")
    
    if result and hasattr(result, "metrics"):
        m = result.metrics
        logger.info(f"  Test Sharpe ratio:   {m.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  Test Sortino ratio:  {m.get('sortino_ratio', 0):.4f}")
        logger.info(f"  Max drawdown:        {m.get('max_drawdown', 0):.2%}")
        logger.info(f"  Win rate:            {m.get('win_rate', 0):.2%}")
        logger.info(f"  Total return:        {m.get('total_return', 0):.2%}")
    
    logger.info("=" * 70)


def main():
    """Main execution entry point."""
    args = parse_args()
    
    # ─── Setup ────────────────────────────────────────────────────
    config = load_config(args.config)
    
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)
    
    device_pref = config.get("project", {}).get("device", "cuda")
    device = get_device(device_pref)
    
    dirs = setup_output_dirs(args.output_dir)
    logger = setup_logger("main", log_file=os.path.join(dirs["logs"], "run.log"))
    
    logger.info("=" * 70)
    logger.info(f"  FinSentNet v0.1.0 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 70)
    
    history = None
    result = None
    metadata = {}
    
    # ─── Data Pipeline ────────────────────────────────────────────
    if args.mode in ("full", "train", "data_only"):
        train_loader, val_loader, test_loader, metadata = run_data_pipeline(
            config, args, logger
        )
        
        if args.mode == "data_only":
            logger.info("Data pipeline complete. Exiting (data_only mode).")
            return
    
    # ─── Model ────────────────────────────────────────────────────
    if args.mode in ("full", "train", "backtest"):
        model = build_model(config, metadata, device, logger)
        
        # Load checkpoint for backtest mode
        if args.mode == "backtest":
            ckpt_path = args.checkpoint or os.path.join(
                dirs["checkpoints"], "finsent_final.pt"
            )
            if not os.path.exists(ckpt_path):
                logger.error(f"Checkpoint not found: {ckpt_path}")
                sys.exit(1)
            
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"  Loaded checkpoint from {ckpt_path}")
            
            # Need data for backtest
            if "test_loader" not in dir():
                train_loader, val_loader, test_loader, metadata = run_data_pipeline(
                    config, args, logger
                )
    
    # ─── Training ─────────────────────────────────────────────────
    if args.mode in ("full", "train"):
        history = run_training(
            model, config, train_loader, val_loader, device, dirs, logger
        )
    
    # ─── Backtesting ──────────────────────────────────────────────
    if args.mode in ("full", "backtest"):
        result = run_backtest(
            model, config, test_loader, metadata, device, dirs, logger
        )
    
    # ─── Summary ──────────────────────────────────────────────────
    print_summary(metadata, history, result, logger)
    
    logger.info("Done.")


if __name__ == "__main__":
    main()
