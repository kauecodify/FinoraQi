"""AutoML: análise de dataset, seleção de modelo por validação cruzada e predição.

Substitui app.ml.automl_engine (inexistente na v1) e app/quantics (dependência inexistente).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_CATEGORIES = 50


def infer_problem_type(y: pd.Series) -> str:
    if y.dtype == object or str(y.dtype) in ("category", "bool"):
        return "classification"
    n_unique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and n_unique <= max(10, int(len(y) * 0.05)):
        return "classification"
    return "regression"


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and df[c].nunique(dropna=True) <= MAX_CATEGORIES]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("nenhuma coluna utilizável como feature")
    return ColumnTransformer(transformers)


def candidates(problem_type: str) -> dict:
    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }


@dataclass
class TrainResult:
    model_id: str
    problem_type: str
    target: str
    features: list[str]
    best_model: str
    leaderboard: list[dict]
    holdout: dict
    pipeline: Pipeline = field(repr=False)


def train(df: pd.DataFrame, target: str, features: list[str] | None = None, problem_type: str | None = None, cv: int = 3) -> TrainResult:
    if target not in df.columns:
        raise ValueError(f"coluna alvo '{target}' não encontrada")
    data = df.dropna(subset=[target]).copy()
    if len(data) < 20:
        raise ValueError("são necessárias pelo menos 20 linhas com alvo preenchido")
    feature_cols = features or [c for c in data.columns if c != target]
    X = data[feature_cols]
    y = data[target]
    numeric, categorical = split_columns(X)
    used = numeric + categorical
    if not used:
        raise ValueError("nenhuma feature numérica ou categórica de baixa cardinalidade")
    X = X[used]
    ptype = problem_type or infer_problem_type(y)

    if ptype == "classification":
        y = y.astype(str)
        min_class = int(y.value_counts().min())
        if y.nunique() < 2:
            raise ValueError("classificação exige pelo menos 2 classes")
        folds = max(2, min(cv, min_class))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring = "accuracy"
        stratify = y if min_class >= 2 else None
    else:
        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna()
        X, y = X[mask], y[mask]
        splitter = KFold(n_splits=max(2, cv), shuffle=True, random_state=42)
        scoring = "r2"
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    board = []
    best_name, best_score = None, -np.inf
    for name, est in candidates(ptype).items():
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", est)])
        scores = cross_val_score(pipe, X_train, y_train, cv=splitter, scoring=scoring)
        mean = float(np.mean(scores))
        board.append({"model": name, "metric": scoring, "cv_mean": round(mean, 4), "cv_std": round(float(np.std(scores)), 4)})
        if mean > best_score:
            best_name, best_score = name, mean

    best = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", candidates(ptype)[best_name])])
    best.fit(X_train, y_train)
    holdout = {scoring: round(float(best.score(X_test, y_test)), 4), "n_test": int(len(y_test))}
    best.fit(X, y)
    board.sort(key=lambda r: r["cv_mean"], reverse=True)
    return TrainResult(str(uuid.uuid4()), ptype, target, used, best_name, board, holdout, best)


def save(result: TrainResult, directory: str) -> str:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{result.model_id}.joblib"
    joblib.dump({"pipeline": result.pipeline, "features": result.features, "problem_type": result.problem_type}, file)
    return str(file)


def load(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("arquivo de modelo inválido")
    return bundle


def predict(bundle: dict, rows: list[dict]) -> list:
    df = pd.DataFrame(rows)
    for col in bundle["features"]:
        if col not in df.columns:
            df[col] = np.nan
    preds = bundle["pipeline"].predict(df[bundle["features"]])
    return [p.item() if hasattr(p, "item") else p for p in preds]
