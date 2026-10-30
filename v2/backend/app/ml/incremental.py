"""Treinamento incremental (online) corrigido.

Diferenças em relação à v1:
- scaler atualizado com partial_fit (não é mais ajustado em uma única amostra);
- classes declaradas no primeiro partial_fit, evitando erro em classes novas;
- categorias desconhecidas mapeadas para um código reservado em vez de quebrar;
- treino em mini-lotes, não amostra a amostra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class CategoryEncoder:
    def __init__(self) -> None:
        self.maps: dict[str, dict[str, int]] = {}

    def fit_update(self, df: pd.DataFrame, columns: list[str]) -> None:
        for col in columns:
            mapping = self.maps.setdefault(col, {})
            for value in df[col].astype(str).unique():
                mapping.setdefault(value, len(mapping) + 1)

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            mapping = self.maps.get(col, {})
            out[col] = out[col].astype(str).map(mapping).fillna(0).astype(float)
        return out


class IncrementalTrainer:
    def __init__(self, problem_type: str, features: list[str], categorical: list[str] | None = None, classes: list | None = None) -> None:
        if problem_type not in ("classification", "regression"):
            raise ValueError("problem_type deve ser classification ou regression")
        self.problem_type = problem_type
        self.features = list(features)
        self.categorical = [c for c in (categorical or []) if c in self.features]
        self.encoder = CategoryEncoder()
        self.scaler = StandardScaler()
        self.classes_ = np.array(sorted(classes)) if classes is not None else None
        self.model = SGDClassifier(loss="log_loss", random_state=42) if problem_type == "classification" else SGDRegressor(random_state=42)
        self.n_seen = 0
        self.history: list[dict] = []

    def _matrix(self, df: pd.DataFrame, update: bool) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"colunas ausentes: {missing}")
        frame = df[self.features]
        if update:
            self.encoder.fit_update(frame, self.categorical)
        frame = self.encoder.transform(frame, self.categorical)
        X = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if update:
            self.scaler.partial_fit(X)
        return self.scaler.transform(X)

    def partial_fit(self, df: pd.DataFrame, y: pd.Series | np.ndarray) -> dict:
        X = self._matrix(df, update=True)
        y_arr = np.asarray(y)
        if self.problem_type == "classification":
            if self.classes_ is None:
                self.classes_ = np.unique(y_arr)
            unknown = set(np.unique(y_arr)) - set(self.classes_.tolist())
            if unknown:
                raise ValueError(f"classes não declaradas: {sorted(unknown)}")
            self.model.partial_fit(X, y_arr, classes=self.classes_)
        else:
            self.model.partial_fit(X, y_arr.astype(float))
        self.n_seen += len(y_arr)
        metrics = self.evaluate(df, y_arr)
        self.history.append({"n_seen": self.n_seen, **metrics})
        return metrics

    def fit_stream(self, df: pd.DataFrame, target: str, batch_size: int = 256, epochs: int = 3) -> dict:
        if self.problem_type == "classification" and self.classes_ is None:
            self.classes_ = np.unique(df[target].to_numpy())
        metrics: dict = {}
        for _ in range(max(1, epochs)):
            for start in range(0, len(df), batch_size):
                chunk = df.iloc[start : start + batch_size]
                metrics = self.partial_fit(chunk, chunk[target])
        return self.evaluate(df, df[target].to_numpy()) if len(df) else metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_seen == 0:
            raise RuntimeError("modelo ainda não treinado")
        return self.model.predict(self._matrix(df, update=False))

    def evaluate(self, df: pd.DataFrame, y: np.ndarray) -> dict:
        pred = self.model.predict(self._matrix(df, update=False))
        if self.problem_type == "classification":
            return {"accuracy": round(float(accuracy_score(y, pred)), 4)}
        y = np.asarray(y, dtype=float)
        r2 = float(r2_score(y, pred)) if len(y) > 1 else 0.0
        return {"r2": round(r2, 4), "mae": round(float(mean_absolute_error(y, pred)), 4)}
