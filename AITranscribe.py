import numpy as np

def extract_features(bars: list) -> dict:
    """
    Convert [open, high, low, close, volume] bars into minimal features:
    - dist_to_ema21_pts: last close - EMA21
    - ema21_slope: average change in EMA21 over last 5 bars
    """
    if len(bars) < 21 + 5:
        raise ValueError("Need at least period+5 bars for EMA slope calculation")

    # Extract closes
    closes = np.array([b['close'] for b in bars], dtype=np.float32)

    # --- Compute EMA21 ---
    ema = np.zeros_like(closes)
    alpha = 2 / (21 + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]

    # --- Current EMA & distance ---
    ema_current = ema[-1]
    close_current = closes[-1]
    ema_distance = close_current - ema_current

    print(close_current,ema_current)

    # --- EMA slope: average change over last 5 bars ---
    ema_slope = (ema[-1] - ema[-6]) / 5.0

    return {
        "dist_to_ema21_pts": float(ema_distance),
        "ema21_slope": float(ema_slope),
    }
