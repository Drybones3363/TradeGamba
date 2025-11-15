import numpy as np

# =========================
# Feature Extraction Methods
# =========================

def extract_features_raw_ohlcv(bars: list, lookback: int = 100) -> dict:
    """
    Extract raw OHLCV data from the last N candles.
    Returns a flat dictionary with keys like:
    - candle_0_open, candle_0_high, ..., candle_0_volume
    - candle_1_open, candle_1_high, ..., candle_1_volume
    - ...
    - candle_49_open, ..., candle_49_volume
    
    Total features: lookback * 5 (e.g., 50 * 5 = 250 features)
    
    This approach lets the neural network learn patterns directly from raw data.
    """
    if len(bars) < lookback:
        raise ValueError(f"Need at least {lookback} bars for raw OHLCV extraction")
    
    # Take the last N bars
    recent_bars = bars[-lookback:]
    
    features = {}
    for i, bar in enumerate(recent_bars):
        features[f"candle_{i}_open"] = float(bar['open'])
        features[f"candle_{i}_high"] = float(bar['high'])
        features[f"candle_{i}_low"] = float(bar['low'])
        features[f"candle_{i}_close"] = float(bar['close'])
        features[f"candle_{i}_volume"] = float(bar.get('volume', 0))
    
    return features


def extract_features(bars: list, ema21_period: int = 21, ema9_period: int = 9) -> dict:
    """
    Extract engineered features based on technical analysis.
    
    Features include:
    - dist_to_ema21_pts: Distance from current close to EMA21
    - ema21_slope: Rate of change of EMA21
    - dist_to_ema9_pts: Distance from current close to EMA9
    - ema9_slope: Rate of change of EMA9
    - recent_volume_ratio: Current volume vs average of last 10 bars
    - body_ratio: Candle body size relative to full range (bullish/bearish strength)
    - upper_wick_ratio: Upper wick size relative to full range
    - lower_wick_ratio: Lower wick size relative to full range
    - close_position: Where close is within the candle's range (0=low, 1=high)
    
    This approach uses domain knowledge to create meaningful features.
    """
    min_bars_needed = max(ema21_period + 5, ema9_period + 5, 10)
    if len(bars) < min_bars_needed:
        raise ValueError(f"Need at least {min_bars_needed} bars for feature calculation")

    # Extract OHLCV arrays
    opens = np.array([b['open'] for b in bars], dtype=np.float32)
    highs = np.array([b['high'] for b in bars], dtype=np.float32)
    lows = np.array([b['low'] for b in bars], dtype=np.float32)
    closes = np.array([b['close'] for b in bars], dtype=np.float32)
    volumes = np.array([b.get('volume', 0) for b in bars], dtype=np.float32)

    # --- Compute EMA21 ---
    ema21 = compute_ema(closes, ema21_period)
    ema21_current = ema21[-1]
    close_current = closes[-1]
    dist_to_ema21 = close_current - ema21_current
    ema21_slope = (ema21[-1] - ema21[-6]) / 5.0

    # --- Compute EMA9 ---
    ema9 = compute_ema(closes, ema9_period)
    ema9_current = ema9[-1]
    dist_to_ema9 = close_current - ema9_current
    ema9_slope = (ema9[-1] - ema9[-6]) / 5.0 if len(ema9) >= 6 else 0.0

    # --- Volume analysis ---
    recent_avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0
    current_volume = volumes[-1]
    volume_ratio = current_volume / recent_avg_volume if recent_avg_volume > 0 else 1.0

    # --- Current candle characteristics ---
    last_bar = bars[-1]
    o, h, l, c = last_bar['open'], last_bar['high'], last_bar['low'], last_bar['close']
    
    candle_range = h - l
    body_size = abs(c - o)
    
    # Ratios (avoid division by zero)
    body_ratio = body_size / candle_range if candle_range > 0 else 0.0
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0.0
    lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0.0
    
    # Where is close within the range? (0 = at low, 1 = at high)
    close_position = (c - l) / candle_range if candle_range > 0 else 0.5

    return {
        "dist_to_ema21_pts": float(dist_to_ema21),
        "ema21_slope": float(ema21_slope),
        "dist_to_ema9_pts": float(dist_to_ema9),
        "ema9_slope": float(ema9_slope),
        "volume_ratio": float(volume_ratio),
        "body_ratio": float(body_ratio),
        "upper_wick_ratio": float(upper_wick_ratio),
        "lower_wick_ratio": float(lower_wick_ratio),
        "close_position": float(close_position),
    }


def extract_features_ema21(bars: list, period: int = 21) -> dict:
    """
    LEGACY: Minimal features for backward compatibility.
    - dist_to_ema21_pts: last close - EMA21
    - ema21_slope: average change in EMA21 over last 5 bars
    
    Recommend using extract_features_engineered() for more comprehensive features.
    """
    if len(bars) < period + 5:
        raise ValueError("Need at least period+5 bars for EMA slope calculation")

    closes = np.array([b['close'] for b in bars], dtype=np.float32)
    ema = compute_ema(closes, period)

    ema_current = ema[-1]
    close_current = closes[-1]
    ema_distance = close_current - ema_current
    ema_slope = (ema[-1] - ema[-6]) / 5.0

    print(close_current, ema_current)

    return {
        "dist_to_ema21_pts": float(ema_distance),
        "ema21_slope": float(ema_slope),
    }


# =========================
# Helper Functions
# =========================

def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average."""
    ema = np.zeros_like(values)
    alpha = 2 / (period + 1)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema