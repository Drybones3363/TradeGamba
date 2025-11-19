#!/usr/bin/env python3
"""
Quick AI Training Script
Reads CSV file and trains the AI on all valid setups
"""

import requests
import json
from datetime import datetime
import AIWebServer

# Configuration
CSV_FILE = "Replays/NQ-09-25.Last.csv"  # Change to your file
WINDOW_BACK = 100  # Bars before entry
LOOKAHEAD_MAX = 600  # Bars after entry to find outcome

# Strategy config (match your JavaScript)
TP_POINTS = 20.0
SL_POINTS = 10.0
TRAIL_TRIGGER = 7.0
TRAIL_OFFSET = 1.0

def is_session_time(timestamp_str):
    """Check if timestamp is during trading session (9:30-15:30 EST)"""
    try:
        # Parse: 20250701 144500
        dt = datetime.strptime(timestamp_str, "%Y%m%d %H%M%S")
        minutes = dt.hour * 60 + dt.minute
        return 600 <= minutes <= 930  # 10:00 to 15:30
    except:
        return True  # If can't parse, include it

def parse_csv(filename):
    """Parse CSV and return list of bars"""
    bars = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line:
                continue
            
            parts = line.split(';')
            if len(parts) < 5:
                continue
            try:
                # Parse: timestamp;open;high;low;close;volume
                dt = datetime.strptime(parts[0], "%Y%m%d %H%M%S")
                timestamp = int(dt.timestamp())
                
                bar = {
                    'time': timestamp,
                    'open': float(parts[1]),
                    'high': float(parts[2]),
                    'low': float(parts[3]),
                    'close': float(parts[4]),
                    'volume': int(parts[5]) if len(parts) > 5 else 1000
                }
                bars.append(bar)
            except Exception as e:
                print(f"Skipping line: {line[:50]}... Error: {e}")
                continue
    
    return bars

def simulate_long(entry, highs, lows):
    """Simulate a long trade"""
    stop = entry - SL_POINTS
    tp = entry + TP_POINTS
    trailed = False
    
    for i in range(min(LOOKAHEAD_MAX, len(highs))):
        hi, lo = highs[i], lows[i]
        
        # Check SL first
        if lo <= stop:
            return stop - entry  # Loss
        
        # Check TP
        if hi >= tp:
            return tp - entry  # Win
        
        # Trail stop
        if not trailed and hi >= entry + TRAIL_TRIGGER:
            stop = entry + TRAIL_OFFSET
            trailed = True
    
    # Timeout
    last_price = (highs[min(LOOKAHEAD_MAX-1, len(highs)-1)] + 
                  lows[min(LOOKAHEAD_MAX-1, len(lows)-1)]) / 2.0
    return last_price - entry

def simulate_short(entry, highs, lows):
    """Simulate a short trade"""
    stop = entry + SL_POINTS
    tp = entry - TP_POINTS
    trailed = False
    
    for i in range(min(LOOKAHEAD_MAX, len(highs))):
        hi, lo = highs[i], lows[i]
        
        # Check SL first
        if hi >= stop:
            return entry - stop  # Loss
        
        # Check TP
        if lo <= tp:
            return entry - tp  # Win
        
        # Trail stop
        if not trailed and lo <= entry - TRAIL_TRIGGER:
            stop = entry - TRAIL_OFFSET
            trailed = True
    
    # Timeout
    last_price = (highs[min(LOOKAHEAD_MAX-1, len(highs)-1)] + 
                  lows[min(LOOKAHEAD_MAX-1, len(lows)-1)]) / 2.0
    return entry - last_price



def trainModel(data):
    bars = data["bars"]
    rewardInfo = data["rewards"]
    entry = float(data["entry"])

    # future highs/lows after the entry bar
    highs = np.array([b['high'] for b in bars[1:]], dtype=np.float32)
    lows  = np.array([b['low']  for b in bars[1:]], dtype=np.float32)

    # features from the full window you sent
    feats = extractFeatures(bars)

    # compute rewards for each possible action at this setup
    actions = [-1, 0, 1]  # short / skip / long
    rewards = [
        learner.step(-1, rewardInfo),
        learner.step( 0, rewardInfo),
        learner.step(+1, rewardInfo),
    ]

    # train on this one setup by duplicating features for each action
    model.train_on_batch([feats, feats, feats], actions, rewards, iters=1)
    model.agent.save(MODEL_FILE+".pkl")

def train_setup(bars, entry_idx):
    """Train AI on one setup"""
    entry = bars[entry_idx]['close']
    
    # Get future bars for simulation
    future_bars = bars[entry_idx + 1:]
    if len(future_bars) < 10:
        return None
    
    highs = [b['high'] for b in future_bars]
    lows = [b['low'] for b in future_bars]
    
    # Simulate both directions
    long_pnl = simulate_long(entry, highs, lows)
    short_pnl = simulate_short(entry, highs, lows)
    
    # Prepare data for server
    history_bars = bars[max(0, entry_idx - WINDOW_BACK):entry_idx + 1]
    
    data = {
        'bars': history_bars,
        'entry': entry,
        'rewards': {
            'long': long_pnl,
            'short': short_pnl,
            'none': 0.0
        }
    }

    trainModel(data)

def main():
    print("=" * 60)
    print("AI TRAINING SCRIPT")
    print("=" * 60)
    
    # Parse CSV
    print(f"\nLoading {CSV_FILE}...")
    bars = parse_csv(CSV_FILE)
    print(f"Loaded {len(bars)} bars during session hours")
    
    if len(bars) < WINDOW_BACK + LOOKAHEAD_MAX + 50:
        print(f"ERROR: Need at least {WINDOW_BACK + LOOKAHEAD_MAX + 50} bars")
        return
    
    # Find valid entry points
    print(f"\nFinding valid setups...")
    valid_entries = []
    for i in range(WINDOW_BACK, len(bars) - LOOKAHEAD_MAX):
        valid_entries.append(i)
    
    print(f"Found {len(valid_entries)} potential setups")
    
    # Train on all setups
    print(f"\nStarting training...")
    print(f"This will take a few minutes...\n")
    
    wins = 0
    losses = 0
    total_pnl = 0
    
    for count, entry_idx in enumerate(valid_entries, 1):
        result = train_setup(bars, entry_idx)
        
        if result is None:
            continue
        
        # Track best trade (long or short)
        best_pnl = max(result['long_pnl'], result['short_pnl'])
        total_pnl += best_pnl
        
        if best_pnl > 5:
            wins += 1
        elif best_pnl < -5:
            losses += 1
        
        # Progress update every 50
        if count % 50 == 0:
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            avg_pnl = total_pnl / count
            print(f"Progress: {count}/{len(valid_entries)} | "
                  f"W/L: {wins}/{losses} ({wr:.1f}%) | "
                  f"Avg P&L: {avg_pnl:.2f} | "
                  f"Score: {result['new_score']:.2f}")
    
    # Final stats
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total setups trained: {len(valid_entries)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    if wins + losses > 0:
        print(f"Win rate: {wins / (wins + losses) * 100:.1f}%")
    print(f"Total P&L: {total_pnl:.2f} points")
    print(f"Average P&L: {total_pnl / len(valid_entries):.2f} points")
    print(f"\nModel saved to {MODEL_FILE}.pkl")
    print("Ready to use!")

if __name__ == "__main__":
    main()