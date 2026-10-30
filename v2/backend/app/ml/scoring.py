"""Score de risco de pagamentos.

Convenção única da v2: 0.0 = risco mínimo, 1.0 = risco máximo.
O núcleo vetorizado usa Numba quando disponível e cai para NumPy puro caso contrário,
com resultados idênticos (testado em tests/test_ml.py).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap if not args or not callable(args[0]) else args[0]


FEATURE_NAMES = (
    "amount_log",
    "night",
    "weekend",
    "method_risk",
    "category_risk",
    "velocity",
    "amount_deviation",
    "user_trust",
)

WEIGHTS = np.array([1.2, 0.8, 0.3, 2.0, 1.2, 0.6, 1.0, -2.0], dtype=np.float64)
BIAS = -2.5

METHOD_RISK = {"pix": 0.25, "debit": 0.15, "credit": 0.35, "boleto": 0.20, "crypto": 0.85}
CATEGORY_RISK = {"payment": 0.20, "transfer": 0.35, "withdrawal": 0.55, "deposit": 0.05, "refund": 0.15}

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.70


def _risk_numpy(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = features @ weights + bias
    return 1.0 / (1.0 + np.exp(-z))


@njit(cache=True, fastmath=False)
def _risk_numba(features, weights, bias):  # pragma: no cover - compiled path
    n, m = features.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        z = bias
        for j in range(m):
            z += features[i, j] * weights[j]
        out[i] = 1.0 / (1.0 + np.exp(-z))
    return out


def risk_scores(features: np.ndarray, use_numba: bool = True) -> np.ndarray:
    arr = np.ascontiguousarray(features, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"esperado matriz (n, {len(FEATURE_NAMES)})")
    if use_numba and NUMBA_AVAILABLE:
        return _risk_numba(arr, WEIGHTS, BIAS)
    return _risk_numpy(arr, WEIGHTS, BIAS)


@dataclass(frozen=True)
class PaymentFeatures:
    amount: float
    payment_method: str
    category: str
    timestamp: datetime
    recent_amounts: tuple[float, ...] = ()
    user_trust: float = 0.5

    def vector(self) -> np.ndarray:
        hour = self.timestamp.hour
        history = np.asarray(self.recent_amounts, dtype=np.float64)
        if history.size >= 2 and history.std() > 0:
            deviation = float(np.clip((self.amount - history.mean()) / history.std() / 3.0, -1.0, 3.0))
        else:
            deviation = 0.0
        return np.array(
            [
                float(np.log10(max(self.amount, 1.0)) / 6.0),
                1.0 if hour < 6 or hour >= 23 else 0.0,
                1.0 if self.timestamp.weekday() >= 5 else 0.0,
                METHOD_RISK.get(self.payment_method, 0.5),
                CATEGORY_RISK.get(self.category, 0.3),
                min(history.size / 10.0, 1.0),
                deviation,
                float(np.clip(self.user_trust, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )


def score_payment(features: PaymentFeatures) -> float:
    return float(risk_scores(features.vector()[None, :])[0])


def risk_level(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "low"
    if score < HIGH_THRESHOLD:
        return "medium"
    return "high"


def decide(score: float, is_anomaly: bool) -> str:
    level = risk_level(score)
    if level == "high":
        return "rejected"
    if level == "medium" or is_anomaly:
        return "review"
    return "approved"


def warmup() -> None:
    risk_scores(np.zeros((1, len(FEATURE_NAMES))))
