"""Previsão de fluxo de caixa (CashFlowModel).

Modelo: regressão linear com tendência + sazonalidade semanal sobre o saldo diário,
quantis P10/P50/P90 a partir da distribuição empírica dos resíduos.
O núcleo de ajuste usa mínimos quadrados do NumPy (determinístico, sem I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


def _design(t: np.ndarray, dow: np.ndarray) -> np.ndarray:
    onehot = np.zeros((t.size, 6))
    for k in range(6):
        onehot[:, k] = (dow == k).astype(float)
    return np.column_stack([np.ones_like(t, dtype=float), t.astype(float), onehot])


@dataclass
class CashFlowModel:
    coef: np.ndarray | None = None
    residual_q: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start: date | None = None
    n_obs: int = 0

    def fit(self, dates: list[date], values: list[float]) -> CashFlowModel:
        if len(dates) != len(values):
            raise ValueError("dates e values devem ter o mesmo tamanho")
        if not dates:
            raise ValueError("histórico vazio")
        order = np.argsort(np.array([d.toordinal() for d in dates]))
        ds = [dates[i] for i in order]
        y = np.asarray(values, dtype=float)[order]
        self.start = ds[0]
        t = np.array([(d - self.start).days for d in ds])
        dow = np.array([d.weekday() for d in ds])
        if len(ds) < 8:
            self.coef = np.zeros(8)
            self.coef[0] = float(y.mean())
            resid = y - y.mean()
        else:
            X = _design(t, dow)
            self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ self.coef
        self.residual_q = tuple(float(q) for q in np.quantile(resid, [0.1, 0.5, 0.9]))
        self.n_obs = len(ds)
        return self

    def predict(self, horizon: int, last_date: date | None = None) -> dict[str, list]:
        if self.coef is None or self.start is None:
            raise RuntimeError("modelo não ajustado")
        if horizon < 1:
            raise ValueError("horizon deve ser >= 1")
        base = last_date or (self.start + timedelta(days=self.n_obs - 1))
        future = [base + timedelta(days=i) for i in range(1, horizon + 1)]
        t = np.array([(d - self.start).days for d in future])
        dow = np.array([d.weekday() for d in future])
        mean = _design(t, dow) @ self.coef
        q10, q50, q90 = self.residual_q
        return {
            "dates": [d.isoformat() for d in future],
            "p10": [round(float(v), 2) for v in mean + q10],
            "p50": [round(float(v), 2) for v in mean + q50],
            "p90": [round(float(v), 2) for v in mean + q90],
        }


def forecast_series(dates: list[date], values: list[float], horizon: int = 7) -> dict[str, list]:
    return CashFlowModel().fit(dates, values).predict(horizon, last_date=max(dates))
