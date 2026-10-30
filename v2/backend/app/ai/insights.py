"""Assistente financeiro: gera insights determinísticos e explicáveis a partir das transações.

Combina agregação (pandas), detecção de anomalias (z robusto), previsão de saldo
(CashFlowModel) e regras de educação financeira. Não depende de serviços externos.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.ml.anomaly import robust_zscore
from app.ml.forecast import forecast_series


def build_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "type", "amount", "category", "description"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
    df["category"] = df["category"].fillna("outros")
    df["signed"] = df["amount"].where(df["type"] == "income", -df["amount"])
    return df


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"income": 0.0, "expense": 0.0, "balance": 0.0, "savings_rate": 0.0, "by_category": {}, "monthly": []}
    income = float(df.loc[df["type"] == "income", "amount"].sum())
    expense = float(df.loc[df["type"] == "expense", "amount"].sum())
    by_cat = df[df["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).round(2).to_dict()
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    for col in ("income", "expense"):
        if col not in monthly:
            monthly[col] = 0.0
    return {
        "income": round(income, 2),
        "expense": round(expense, 2),
        "balance": round(income - expense, 2),
        "savings_rate": round((income - expense) / income, 4) if income else 0.0,
        "by_category": {k: float(v) for k, v in by_cat.items()},
        "monthly": [
            {"month": r["month"], "income": round(float(r["income"]), 2), "expense": round(float(r["expense"]), 2)}
            for _, r in monthly.iterrows()
        ],
    }


def generate_insights(rows: list[dict], horizon: int = 30) -> dict:
    df = build_frame(rows)
    summary = summarize(df)
    insights: list[dict] = []
    if df.empty:
        insights.append({"level": "info", "title": "Sem dados", "message": "Registre transações para receber recomendações."})
        return {"summary": summary, "insights": insights, "forecast": None}

    rate = summary["savings_rate"]
    if summary["income"] and rate < 0:
        insights.append(
            {"level": "alert", "title": "Gastos acima da renda", "message": f"Despesas superam a renda em R$ {-summary['balance']:.2f}."}
        )
    elif summary["income"] and rate < 0.1:
        insights.append(
            {
                "level": "warning",
                "title": "Poupança baixa",
                "message": f"Taxa de poupança de {rate:.0%}. A referência saudável é 10% a 20%.",
            }
        )
    elif summary["income"]:
        insights.append({"level": "success", "title": "Boa taxa de poupança", "message": f"Você está poupando {rate:.0%} da renda."})

    if summary["by_category"] and summary["expense"]:
        top, value = next(iter(summary["by_category"].items()))
        share = value / summary["expense"]
        if share > 0.35:
            insights.append({"level": "warning", "title": f"Concentração em {top}", "message": f"{share:.0%} das despesas estão em {top}."})

    expenses = df[df["type"] == "expense"].sort_values("date")
    history: list[float] = []
    for _, row in expenses.iterrows():
        z = robust_zscore(row["amount"], history)
        if z > 3.5:
            insights.append(
                {
                    "level": "alert",
                    "title": "Despesa atípica",
                    "message": f"R$ {row['amount']:.2f} em {row['date'].date()} ({row['description'] or row['category']}).",
                }
            )
        history.append(row["amount"])

    daily = df.groupby(df["date"].dt.date)["signed"].sum().cumsum()
    forecast = None
    if len(daily) >= 2:
        dates: list[date] = list(daily.index)
        forecast = forecast_series(dates, daily.tolist(), horizon=horizon)
        if forecast["p50"][-1] < 0:
            insights.append(
                {
                    "level": "alert",
                    "title": "Risco de saldo negativo",
                    "message": f"Projeção mediana de R$ {forecast['p50'][-1]:.2f} em {horizon} dias.",
                }
            )
    return {"summary": summary, "insights": insights, "forecast": forecast}
