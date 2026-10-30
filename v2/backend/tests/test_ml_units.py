from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml import automl_engine, credit_risk, evolution, scoring
from app.ml.anomaly import detect_anomaly, robust_zscore
from app.ml.forecast import CashFlowModel, forecast_series
from app.ml.incremental import IncrementalTrainer


def test_numba_and_numpy_agree():
    X = np.random.default_rng(1).random((500, len(scoring.FEATURE_NAMES)))
    assert np.allclose(scoring.risk_scores(X, True), scoring.risk_scores(X, False))
    with pytest.raises(ValueError):
        scoring.risk_scores(np.zeros((2, 3)))


def test_score_monotonic_in_risk_factors():
    base = dict(amount=100.0, category="payment", timestamp=datetime(2026, 3, 4, 14), user_trust=0.8)
    low = scoring.score_payment(scoring.PaymentFeatures(payment_method="debit", **base))
    high = scoring.score_payment(scoring.PaymentFeatures(payment_method="crypto", **base))
    assert high > low
    assert scoring.decide(0.1, False) == "approved"
    assert scoring.decide(0.1, True) == "review"
    assert scoring.decide(0.9, False) == "rejected"


def test_anomaly():
    assert robust_zscore(10, [1, 2]) == 0.0
    flagged, reasons = detect_anomaly(10_000, [100, 101, 99, 100, 102], "pix")
    assert flagged and reasons


def test_credit_risk_bounds():
    good = credit_risk.default_probability(30, 0, 50_000, 30)
    bad = credit_risk.default_probability(1, 10, 100, 11)
    assert good < bad
    assert credit_risk.credit_score(0) == 900 and credit_risk.credit_score(1) == 300


def test_forecast_trend_and_weekly():
    start = date(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(60)]
    values = [1000 + 10 * i + (50 if d.weekday() == 0 else 0) for i, d in enumerate(dates)]
    out = forecast_series(dates, values, horizon=7)
    assert len(out["p50"]) == 7
    assert out["p50"][-1] > values[-1]
    assert all(a <= b <= c for a, b, c in zip(out["p10"], out["p50"], out["p90"], strict=False))
    with pytest.raises(RuntimeError):
        CashFlowModel().predict(3)


def _dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"renda": rng.normal(5000, 1500, n), "idade": rng.integers(18, 70, n), "regiao": rng.choice(["sul", "norte", "leste"], n)}
    )
    df["inadimplente"] = np.where(df["renda"] < 4500, "sim", "nao")
    df["gasto"] = df["renda"] * 0.6 + rng.normal(0, 100, n)
    return df


def test_automl_classification_and_regression(tmp_path):
    df = _dataset()
    clf = automl_engine.train(df.drop(columns=["gasto"]), "inadimplente")
    assert clf.problem_type == "classification" and clf.holdout["accuracy"] > 0.8
    reg = automl_engine.train(df.drop(columns=["inadimplente"]), "gasto")
    assert reg.problem_type == "regression" and reg.holdout["r2"] > 0.8
    bundle = automl_engine.load(automl_engine.save(reg, str(tmp_path)))
    preds = automl_engine.predict(bundle, [{"renda": 5000, "idade": 30, "regiao": "desconhecida"}])
    assert 2500 < preds[0] < 3500


def test_incremental_handles_new_categories_and_classes():
    df = _dataset(300)
    t = IncrementalTrainer("classification", ["renda", "idade", "regiao"], ["regiao"])
    metrics = t.fit_stream(df, "inadimplente", batch_size=32, epochs=3)
    assert metrics["accuracy"] > 0.8
    unseen = pd.DataFrame({"renda": [9000], "idade": [40], "regiao": ["oeste"]})
    assert t.predict(unseen)[0] == "nao"
    reg = IncrementalTrainer("regression", ["renda", "idade"])
    assert reg.fit_stream(df, "gasto", batch_size=50, epochs=5)["r2"] > 0.8
    with pytest.raises(ValueError):
        IncrementalTrainer("cluster", ["a"])


def test_evolution_returns_integer_params():
    out = evolution.evolve(_dataset(120).drop(columns=["gasto"]), "inadimplente", generations=2, population=4)
    assert isinstance(out["best_params"]["n_estimators"], int)
    assert out["best_score"] > 0.7
