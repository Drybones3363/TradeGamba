from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle, os


import AITranscribe

# =========================
# Strategy & Simulation
# =========================

@dataclass
class StrategyConfig:
    take_profit_pts: float = 30.0        # TP in points
    stop_loss_pts: float = 20.0          # SL in points
    trail_trigger_pts: float = 10.0       # when price goes +10 from entry, trail
    trail_to_offset_pts: float = 2.0     # move stop to entry +2 (or -2 for short)
    max_bars_ahead: int = 150            # max bars to hold (failsafe timeout)

def simulate_trade_long(entry: float, highs: np.ndarray, lows: np.ndarray, cfg: StrategyConfig) -> float:
    """
    Returns P&L in points for a LONG using the trailing-breakeven (+2) once +5 is reached.
    highs/lows are future bars after entry (2-min bars).
    """
    stop = entry - cfg.stop_loss_pts
    tp = entry + cfg.take_profit_pts
    trailed = False

    for i in range(min(cfg.max_bars_ahead, len(highs))):
        hi, lo = highs[i], lows[i]

        # check TP/SL intrabar (assume worst-case: SL fills before TP if both touched same bar)
        # 1) SL
        if lo <= stop:
            return stop - entry  # negative

        # 2) TP
        if hi >= tp:
            return tp - entry    # positive

        # 3) trailing trigger
        if not trailed and hi - entry >= cfg.trail_trigger_pts:
            stop = entry + cfg.trail_to_offset_pts
            trailed = True

    # Timeout exit at last close
    last_price = (highs[min(cfg.max_bars_ahead, len(highs))-1] + lows[min(cfg.max_bars_ahead, len(highs))-1]) / 2.0
    return last_price - entry

def simulate_trade_short(entry: float, highs: np.ndarray, lows: np.ndarray, cfg: StrategyConfig) -> float:
    """
    Returns P&L in points for a SHORT (mirror logic).
    """
    stop = entry + cfg.stop_loss_pts
    tp = entry - cfg.take_profit_pts
    trailed = False

    for i in range(min(cfg.max_bars_ahead, len(highs))):
        hi, lo = highs[i], lows[i]

        # SL first (worst-case order)
        if hi >= stop:
            return entry - stop

        # TP
        if lo <= tp:
            return entry - tp

        # trailing trigger
        if not trailed and entry - lo >= cfg.trail_trigger_pts:
            stop = entry - cfg.trail_to_offset_pts
            trailed = True

    last_price = (highs[min(cfg.max_bars_ahead, len(highs))-1] + lows[min(cfg.max_bars_ahead, len(highs))-1]) / 2.0
    return entry - last_price

# =========================
# Feature Handling
# =========================

"""
You can feed any features you compute from your 2-min chart (VWAP distance,
21 EMA/300 SMA slope, proximity to 15-min range edges, wick/volume/momentum slowdown, etc.)
Provide them as a dict of floats. Register the keys below so the model knows the order.
"""

FEATURE_KEYS = []
for i in range(100):
    FEATURE_KEYS.append(f"candle_{i}_open")
    FEATURE_KEYS.append(f"candle_{i}_high")
    FEATURE_KEYS.append(f"candle_{i}_low")
    FEATURE_KEYS.append(f"candle_{i}_close")
    FEATURE_KEYS.append(f"candle_{i}_volume")

def features_to_vector(feat: Dict[str, float]) -> np.ndarray:
    vec = np.array([feat.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
    return vec

# =========================
# RL Agent (policy + value)
# =========================

class TinyActorCritic:
    """
    Simple 2-layer MLP for:
      - Policy π(a|s) over actions: {-1: short, 0: skip, +1: long}
      - Value V(s) (baseline)
    Trains with REINFORCE + value regression.
    """
    def __init__(self, n_features: int, hidden: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(n_features, hidden))
        self.b1 = np.zeros((hidden,))
        self.Wp = rng.normal(0, 0.1, size=(hidden, 3))  # policy head
        self.bp = np.zeros((3,))
        self.Wv = rng.normal(0, 0.1, size=(hidden, 1))  # value head
        self.bv = np.zeros((1,))
        self.lr = 1e-3

    def save(self, path="model.pkl"):
        data = {
            "W1": self.W1, "b1": self.b1,
            "Wp": self.Wp, "bp": self.bp,
            "Wv": self.Wv, "bv": self.bv,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path="model.pkl"):
        if not os.path.exists(path):
            print("⚠️ No saved model found, skipping load.")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.W1 + self.b1)                  # (B, H)
        logits = h @ self.Wp + self.bp                      # (B, 3)
        # stable softmax
        z = logits - logits.max(axis=1, keepdims=True)
        expz = np.exp(z)
        probs = expz / expz.sum(axis=1, keepdims=True)
        value = h @ self.Wv + self.bv                       # (B, 1)
        return h, probs, value.squeeze(-1)

    def act(self, x: np.ndarray) -> Tuple[int, float]:
        x = x.reshape(1, -1)
        _, probs, _ = self._forward(x)
        p = probs[0]
        action_idx = np.random.choice(3, p=p)
        # map index->action
        action_map = {0: -1, 1: 0, 2: 1}
        return action_map[action_idx], p[action_idx]

    def predict_score(self, x: np.ndarray) -> float:
        """
        Convert policy preference to a signed score [-100, 100].
        Uses difference between Long and Short probabilities, scaled by confidence.
        """
        x = x.reshape(1, -1)
        _, probs, _ = self._forward(x)
        p_short, p_skip, p_long = probs[0]
        pref = p_long - p_short  # [-1..1]
        # Confidence down-weights when skip dominates
        conf = 1.0 - p_skip      # [0..1]
        score = 100.0 * pref * conf
        return float(np.clip(score, -100.0, 100.0))

    def train_batch(self, X: np.ndarray, actions: np.ndarray, rewards: np.ndarray, iters: int = 1):
        """
        One-step REINFORCE with value baseline and simple MSE on value.
        actions in {-1,0,1} mapped to indices {0,1,2}.
        rewards are realized P&L in points (can scale to R if preferred).
        """
        idx_map = {-1:0, 0:1, 1:2}
        A = np.array([idx_map[a] for a in actions], dtype=np.int64)
        for _ in range(iters):
            # forward pass
            h, probs, values = self._forward(X)   # shapes: (B,H), (B,3), (B,)
            # advantages
            adv = rewards - values                # (B,)

            # policy loss: -log π(a|s) * adv
            logp = np.log(np.take_along_axis(probs, A[:,None], axis=1).squeeze(1) + 1e-8)
            L_policy = -(logp * adv).mean()

            # value loss
            L_value = ((values - rewards)**2).mean() * 0.5

            # total
            L = L_policy + L_value * 0.5

            # Backprop (manual for clarity; small net)
            B = X.shape[0]
            # dL/dvalues
            dvalues = (values - rewards) * 0.5 * 2.0  # = (values - rewards)
            # dL/dlogits via softmax cross-entropy with advantage
            dlogp = -(adv / B)                        # derivative wrt logπ(a)
            dlogits = np.zeros_like(probs)
            dlogits[np.arange(B), A] = dlogp
            # convert from d/dlogπ to d/dlogits for softmax:
            # For softmax: dL/dz = (p - y)*scalar, but with advantage we can use
            # policy gradient trick: grad = -adv * grad(logπ(a)) which equals:
            # grad wrt logits = (probs); then subtract 1 on taken action row.
            # Here we already put mass on taken action via dlogits, so adjust:
            dlogits = probs / B + dlogits            # spread + focus on action
            dlogits[np.arange(B), A] -= (1.0 / B)

            # Back to heads
            dWp = h.T @ dlogits
            dbp = dlogits.sum(axis=0)

            dV = dvalues[:,None]                     # (B,1)
            dWv = h.T @ dV
            dbv = dV.sum(axis=0)

            # Into shared hidden
            dh_from_policy = dlogits @ self.Wp.T
            dh_from_value  = dV @ self.Wv.T
            dh = dh_from_policy + dh_from_value      # (B,H)
            dh *= (1 - h**2)                         # tanh'

            dW1 = X.T @ dh
            db1 = dh.sum(axis=0)

            # SGD update
            for param, grad in [
                (self.W1, dW1), (self.b1, db1),
                (self.Wp, dWp), (self.bp, dbp),
                (self.Wv, dWv), (self.bv, dbv),
            ]:
                param -= self.lr * grad

# =========================
# Environment / Reward
# =========================

class SetupEnv:
    """
    One-step environment:
      - State: feature vector for a setup at time t (pre-entry)
      - Action: {-1 short, 0 skip, +1 long}
      - Reward: realized P&L (points) from simulator using your stop/TP/trailing
    """
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def step(self, action: int, rewards: dict) -> float:
        print(rewards)
        if action == 0:
            return rewards['none']
        elif action == 1:
            return rewards['long']
        elif action == -1:
            return rewards['short']
        else:
            raise ValueError("Action must be in {-1, 0, 1}")

# =========================
# Scoring API
# =========================

class ScoringModel:
    """
    Wraps the RL agent to offer a single call:
      score = [-100..100] for given features.
    """
    def __init__(self, feature_keys: List[str] = FEATURE_KEYS, hidden: int = 32, seed: int = 42):
        self.feature_keys = feature_keys
        self.agent = TinyActorCritic(n_features=len(feature_keys), hidden=hidden, seed=seed)

    def score_setup(self, feature_dict: Dict[str, float]) -> float:
        x = features_to_vector(feature_dict)
        return self.agent.predict_score(x)

    def choose_action(self, feature_dict: Dict[str, float]) -> int:
        x = features_to_vector(feature_dict)
        act, _ = self.agent.act(x)
        return act

    def train_on_batch(self, feature_dicts: List[Dict[str, float]], actions: List[int], rewards: List[float], iters: int = 1):
        X = np.stack([features_to_vector(fd) for fd in feature_dicts], axis=0)
        A = np.array(actions, dtype=np.int64)
        R = np.array(rewards, dtype=np.float32)
        self.agent.train_batch(X, A, R, iters=iters)







