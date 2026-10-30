"""Modelo de score de crédito (300-900) treinado em dados sintéticos determinísticos.

Substitui o ML_V1 (Celery + Redis + joblib carregado no import). O modelo é treinado
sob demanda, em memória, com semente fixa, e pode ser persistido em models_dir.
"""

import threading

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("on_time_ratio", "late_payments", "total_paid_log", "history_len", "income_ratio")

_lock = threading.Lock()
_model: Pipeline | None = None


def _synthetic(n: int = 4000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    on_time = rng.beta(5, 2, n)
    late = rng.poisson(1.5, n).astype(float)
    paid_log = rng.normal(3.5, 0.8, n)
    hist = rng.integers(0, 60, n).astype(float)
    income = rng.uniform(0.05, 0.9, n)
    logit = -2.5 * on_time + 0.6 * late - 0.5 * (paid_log - 3.5) - 0.02 * hist + 2.0 * income + 0.8
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return np.column_stack([on_time, late, paid_log, hist, income]), default


def get_model() -> Pipeline:
    global _model
    with _lock:
        if _model is None:
            X, y = _synthetic()
            _model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
        return _model


def default_probability(on_time: int, late: int, total_paid: float, history_len: int, income_ratio: float = 0.3) -> float:
    total = on_time + late
    ratio = on_time / total if total else 0.5
    x = np.array([[ratio, late, np.log10(max(total_paid, 1.0)), history_len, income_ratio]], dtype=float)
    return float(get_model().predict_proba(x)[0, 1])


def credit_score(probability_of_default: float) -> int:
    return int(round(900 - 600 * float(np.clip(probability_of_default, 0.0, 1.0))))


def score_band(score: int) -> str:
    if score >= 750:
        return "excelente"
    if score >= 650:
        return "bom"
    if score >= 500:
        return "regular"
    return "baixo"
