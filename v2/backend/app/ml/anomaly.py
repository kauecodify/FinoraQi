import numpy as np


def robust_zscore(value: float, history: list[float] | tuple[float, ...]) -> float:
    arr = np.asarray(history, dtype=np.float64)
    if arr.size < 3:
        return 0.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad == 0:
        return 0.0 if value == median else 10.0
    return 0.6745 * (value - median) / mad


def detect_anomaly(amount: float, history: list[float] | tuple[float, ...], method: str, threshold: float = 3.5) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    z = robust_zscore(amount, history)
    if abs(z) > threshold:
        reasons.append(f"valor fora do padrão (z robusto={z:.2f})")
    if method == "crypto" and amount > 10_000:
        reasons.append("cripto acima de R$ 10.000")
    if len(history) >= 5 and amount > 5 * max(history):
        reasons.append("valor 5x acima do maior histórico")
    return bool(reasons), reasons
