#!/usr/bin/env python3
"""
Training script for AI that only uses scenarios where entry price is between EMA 9 and EMA 21.
Uses existing learningAI.py classes and AITranscribe.py feature extraction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import sys
import os

# Import your existing modules
from learningAI import (
    StrategyConfig, 
    SetupEnv, 
    ScoringModel, 
    simulate_trade_long, 
    simulate_trade_short,
    FEATURE_KEYS
)
from AITranscribe import extract_features_ema21


def calculate_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate EMA for a given period
    """
    ema = np.zeros_like(closes)
    alpha = 2 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
    return ema


def load_bars_from_csv(filepath: str) -> List[Dict]:
    """
    Load OHLCV data from CSV file
    Expected format: time,open,high,low,close,volume
    Or: YYYYMMDD HHmmss;open;high;low;close;volume
    """
    bars = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try semicolon delimiter first (like your NQ data)
        if ';' in line:
            parts = line.split(';')
        else:
            parts = line.split(',')
        
        if len(parts) < 5:
            continue
            
        try:
            # Parse timestamp
            time_str = parts[0].strip()
            
            # Convert YYYYMMDD HHmmss to unix timestamp
            if len(time_str) == 15 and ' ' in time_str:  # "20250910 015400" format
                from datetime import datetime
                dt = datetime.strptime(time_str, "%Y%m%d %H%M%S")
                timestamp = int(dt.timestamp())
            else:
                # Assume it's already a timestamp
                timestamp = int(float(time_str))
            
            bar = {
                'time': timestamp,
                'open': float(parts[1]),
                'high': float(parts[2]),
                'low': float(parts[3]),
                'close': float(parts[4]),
                'volume': float(parts[5]) if len(parts) > 5 else 0
            }
            bars.append(bar)
        except (ValueError, IndexError) as e:
            print(f"Skipping line due to parse error: {line[:50]}... Error: {e}")
            continue
    
    print(f"Loaded {len(bars)} bars from {filepath}")
    return bars


def filter_ema_entry_scenarios(bars: List[Dict], window_size: int = 100) -> List[Tuple[int, List[Dict]]]:
    """
    Filter bars to find scenarios where entry price is between EMA 9 and EMA 21.
    
    Returns:
        List of (entry_index, bars_window) tuples where:
        - entry_index: the index in the original bars array
        - bars_window: window of bars leading up to and including the entry point
    """
    if len(bars) < window_size + 50:  # Need extra bars for future outcome
        print(f"Not enough bars. Need at least {window_size + 50}, got {len(bars)}")
        return []
    
    # Calculate EMAs for all bars
    closes = np.array([b['close'] for b in bars])
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    
    valid_scenarios = []
    
    # Start after we have enough history for the window
    # End before we run out of lookahead bars
    for i in range(window_size, len(bars) - 150):  # Leave 150 bars for outcome simulation
        entry_price = bars[i]['close']
        ema9_val = ema9[i]
        ema21_val = ema21[i]
        
        # Check if price is between EMAs
        is_between = (
            (entry_price >= ema9_val and entry_price <= ema21_val) or
            (entry_price <= ema9_val and entry_price >= ema21_val)
        )
        
        if is_between:
            # Extract the window of bars leading up to this point
            start_idx = max(0, i - window_size + 1)
            bars_window = bars[start_idx:i+1]
            valid_scenarios.append((i, bars_window))
    
    print(f"Found {len(valid_scenarios)} valid EMA entry scenarios out of {len(bars)} total bars")
    print(f"Percentage: {len(valid_scenarios)/len(bars)*100:.2f}%")
    
    return valid_scenarios


def prepare_training_batch(
    bars: List[Dict], 
    scenario_indices: List[Tuple[int, List[Dict]]], 
    cfg: StrategyConfig
) -> Tuple[List[Dict], List[int], List[float]]:
    """
    Prepare a batch of training data from valid scenarios.
    
    Returns:
        feature_dicts: List of feature dictionaries for each scenario
        actions: List of actions taken (we'll train all 3 actions per scenario)
        rewards: List of rewards for each action
    """
    feature_dicts = []
    actions = []
    rewards = []
    
    for entry_idx, bars_window in scenario_indices:
        # Extract features from the window
        try:
            feats = extract_features_ema21(bars_window)
        except ValueError as e:
            print(f"Skipping scenario at index {entry_idx}: {e}")
            continue
        
        # Get entry price and future bars
        entry_price = bars[entry_idx]['close']
        
        # Get future bars for simulation (up to max_bars_ahead)
        future_bars = bars[entry_idx+1 : entry_idx+1+cfg.max_bars_ahead]
        if len(future_bars) < 10:  # Need some minimum future bars
            continue
            
        future_highs = np.array([b['high'] for b in future_bars], dtype=np.float32)
        future_lows = np.array([b['low'] for b in future_bars], dtype=np.float32)
        
        # Simulate outcomes for each action
        # Long action
        pnl_long = simulate_trade_long(entry_price, future_highs, future_lows, cfg)
        
        # Short action
        pnl_short = simulate_trade_short(entry_price, future_highs, future_lows, cfg)
        
        # Skip action (no reward)
        pnl_skip = 0.0
        
        # Add all three actions for this scenario
        for action, reward in [(-1, pnl_short), (0, pnl_skip), (1, pnl_long)]:
            feature_dicts.append(feats)
            actions.append(action)
            rewards.append(reward)
    
    return feature_dicts, actions, rewards


def train_model_on_ema_scenarios(
    csv_filepath: str,
    model_save_path: str = "model_ema_filtered.pkl",
    num_epochs: int = 10,
    batch_iterations: int = 3
):
    """
    Main training function that:
    1. Loads data from CSV
    2. Filters for EMA 9-21 entry scenarios
    3. Trains the model on these scenarios
    """
    print("="*70)
    print("Training AI on EMA 9-21 Entry Scenarios")
    print("="*70)
    
    # Initialize components
    cfg = StrategyConfig()
    learner = SetupEnv(cfg)
    model = ScoringModel()
    
    # Try to load existing model
    if os.path.exists(model_save_path):
        print(f"\nLoading existing model from {model_save_path}")
        model.agent.load(model_save_path)
    else:
        print("\nStarting with fresh model")
    
    # Load data
    print(f"\nLoading data from {csv_filepath}...")
    bars = load_bars_from_csv(csv_filepath)
    
    if len(bars) < 200:
        print("ERROR: Not enough bars loaded. Need at least 200 bars.")
        return
    
    # Filter for valid EMA scenarios
    print("\nFiltering for EMA entry scenarios...")
    valid_scenarios = filter_ema_entry_scenarios(bars, window_size=100)
    
    if len(valid_scenarios) == 0:
        print("ERROR: No valid EMA scenarios found in the data.")
        return
    
    print(f"\nPreparing training data from {len(valid_scenarios)} scenarios...")
    feature_dicts, actions, rewards = prepare_training_batch(bars, valid_scenarios, cfg)
    
    if len(feature_dicts) == 0:
        print("ERROR: No valid training samples could be prepared.")
        return
    
    print(f"\nTotal training samples: {len(feature_dicts)}")
    print(f"  - Long actions: {actions.count(1)}")
    print(f"  - Short actions: {actions.count(-1)}")
    print(f"  - Skip actions: {actions.count(0)}")
    
    # Calculate some statistics on rewards
    rewards_array = np.array(rewards)
    print(f"\nReward statistics:")
    print(f"  - Mean: {rewards_array.mean():.2f} points")
    print(f"  - Std: {rewards_array.std():.2f} points")
    print(f"  - Min: {rewards_array.min():.2f} points")
    print(f"  - Max: {rewards_array.max():.2f} points")
    print(f"  - Positive outcomes: {(rewards_array > 0).sum()} ({(rewards_array > 0).sum()/len(rewards_array)*100:.1f}%)")
    
    # Train the model
    print(f"\n{'='*70}")
    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train on the full batch
        model.train_on_batch(
            feature_dicts, 
            actions, 
            rewards, 
            iters=batch_iterations
        )
        
        # Sample a few scenarios and show predicted scores
        if (epoch + 1) % 2 == 0:  # Every 2 epochs
            print(f"  Sample predictions:")
            sample_indices = np.random.choice(len(feature_dicts)//3, min(5, len(feature_dicts)//3), replace=False)
            for idx in sample_indices:
                feats = feature_dicts[idx*3]  # Get one of each action trio
                score = model.score_setup(feats)
                actual_long = rewards[idx*3 + 2]   # Long is at index +2
                actual_short = rewards[idx*3]      # Short is at index +0
                print(f"    Scenario: Score={score:+.1f}, Actual Long={actual_long:+.1f}pts, Actual Short={actual_short:+.1f}pts")
        
        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Saving model...")
            model.agent.save(model_save_path)
    
    # Final save
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}\n")
    print(f"Saving final model to {model_save_path}")
    model.agent.save(model_save_path)
    
    # Show some final statistics
    print("\nFinal Model Evaluation on Training Data:")
    sample_size = min(20, len(feature_dicts)//3)
    sample_indices = np.random.choice(len(feature_dicts)//3, sample_size, replace=False)
    
    scores = []
    actuals_long = []
    actuals_short = []
    
    for idx in sample_indices:
        feats = feature_dicts[idx*3]
        score = model.score_setup(feats)
        scores.append(score)
        actuals_long.append(rewards[idx*3 + 2])
        actuals_short.append(rewards[idx*3])
    
    print(f"\nSample of {sample_size} scenarios:")
    print(f"  Model scores - Mean: {np.mean(scores):+.1f}, Std: {np.std(scores):.1f}")
    print(f"  Actual Long P&L - Mean: {np.mean(actuals_long):+.1f}, Std: {np.std(actuals_long):.1f}")
    print(f"  Actual Short P&L - Mean: {np.mean(actuals_short):+.1f}, Std: {np.std(actuals_short):.1f}")
    
    return model


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the NQ data in your Replays folder
        csv_file = "Replays/NQ-09-25.Last.csv"
    
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}")
        print("\nUsage: python train_ema_filtered.py [path_to_csv]")
        print(f"  Default CSV: {csv_file}")
        sys.exit(1)
    
    # Train the model
    trained_model = train_model_on_ema_scenarios(
        csv_filepath=csv_file,
        model_save_path="model_ema_filtered.pkl",
        num_epochs=20,
        batch_iterations=5
    )
    
    print("\n" + "="*70)
    print("Training script completed successfully!")
    print("="*70)