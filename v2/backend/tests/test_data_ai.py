import io

import pandas as pd

from app.ai.categorizer import get_categorizer
from app.ai.insights import generate_insights
from app.processing.spreadsheet import clean

CSV = "Data;Descrição;Valor (R$);Tipo\n01/02/2026;Mercado;1.234,50;despesa\n01/02/2026;Mercado;1.234,50;despesa\n03/02/2026;Salário;R$ 5.000,00;receita\n05/02/2026;Uber;;despesa\n"  # noqa: E501


def test_clean_brazilian_formats():
    df = pd.read_csv(io.StringIO(CSV), sep=";")
    out, report = clean(df)
    assert list(out.columns) == ["data", "descricao", "valor_r", "tipo"]
    assert report["duplicates_removed"] == 1
    assert out["valor_r"].dtype.kind == "f" and out["valor_r"].iloc[0] == 1234.5
    assert str(out["data"].dtype).startswith("datetime64")
    assert report["filled"] == {"valor_r": 1}


def test_data_endpoints(client, auth):
    files = {"file": ("extrato.csv", CSV.encode("utf-8"), "text/csv")}
    r = client.post("/api/v1/data/profile", files=files, headers=auth)
    assert r.status_code == 200 and r.json()["rows"] == 4
    r = client.post("/api/v1/data/clean", files=files, data={"fill_strategy": "zero"}, headers=auth)
    assert r.json()["report"]["rows_out"] == 3
    r = client.post("/api/v1/data/aggregate", files=files, data={"group_by": "tipo", "metrics": '{"valor_r": "sum"}'}, headers=auth)
    assert r.status_code == 200 and r.json()["rows"] == 2
    r = client.post("/api/v1/data/export", files=files, data={"fmt": "xlsx"}, headers=auth)
    assert r.status_code == 200 and r.content[:2] == b"PK"
    bad = client.post("/api/v1/data/profile", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=auth)
    assert bad.status_code == 400
    bad = client.post("/api/v1/data/aggregate", files=files, data={"group_by": "tipo", "metrics": '{"valor_r": "drop"}'}, headers=auth)
    assert bad.status_code == 400


def test_categorizer_and_feedback(client, auth):
    preds = get_categorizer().predict(["Netflix mensal", "Farmácia Drogasil", "Aluguel março"])
    assert [p["category"] for p in preds] == ["lazer", "saude", "moradia"]
    assert (
        client.post("/api/v1/ai/categorize/feedback", json={"description": "x", "category": "inexistente"}, headers=auth).status_code == 400
    )
    assert (
        client.post("/api/v1/ai/categorize", json={"descriptions": ["ifood jantar"]}, headers=auth).json()[0]["category"] == "alimentacao"
    )


def test_insights_engine():
    rows = [
        {"date": f"2026-01-{d:02d}", "type": "expense", "amount": 50, "category": "alimentacao", "description": "mercado"}
        for d in range(1, 21)
    ]
    rows.append({"date": "2026-01-21", "type": "expense", "amount": 5000, "category": "compras", "description": "notebook"})
    rows.append({"date": "2026-01-01", "type": "income", "amount": 3000, "category": "salario", "description": "salario"})
    out = generate_insights(rows, horizon=10)
    titles = [i["title"] for i in out["insights"]]
    assert "Gastos acima da renda" in titles and "Despesa atípica" in titles
    assert len(out["forecast"]["p50"]) == 10
    assert generate_insights([])["insights"][0]["title"] == "Sem dados"


def test_insights_and_forecast_endpoints(client, auth):
    client.post(
        "/api/v1/transactions", json={"type": "income", "amount": "4000", "category": "salario", "date": "2026-01-01"}, headers=auth
    )
    client.post(
        "/api/v1/transactions", json={"type": "expense", "amount": "500", "description": "aluguel", "date": "2026-01-05"}, headers=auth
    )
    r = client.get("/api/v1/ai/insights", headers=auth)
    assert r.status_code == 200 and r.json()["summary"]["balance"] == 3500.0
    r = client.post(
        "/api/v1/ai/forecast", json={"dates": ["2026-01-01", "2026-01-02", "2026-01-03"], "values": [1, 2, 3], "horizon": 2}, headers=auth
    )
    assert len(r.json()["p50"]) == 2
