import time

import numpy as np
import pandas as pd


def _csv(n=120):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "grupo": rng.choice(["a", "b"], n)})
    df["y"] = 3 * df["x1"] - 2 * df["x2"] + rng.normal(0, 0.1, n)
    df["classe"] = np.where(df["x1"] > 0, "alto", "baixo")
    return df.to_csv(index=False).encode()


def test_train_predict_list_download(client, auth):
    files = {"file": ("d.csv", _csv(), "text/csv")}
    assert client.post("/api/v1/automl/analyze", files=files, headers=auth).status_code == 200
    r = client.post("/api/v1/automl/train", files=files, data={"target": "y", "features": "x1,x2,grupo"}, headers=auth)
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]
    assert r.json()["holdout"]["r2"] > 0.9
    pred = client.post(
        "/api/v1/automl/predict", json={"model_id": model_id, "rows": [{"x1": 1, "x2": 0, "grupo": "a"}]}, headers=auth
    ).json()
    assert 2 < pred["predictions"][0] < 4
    assert any(m["model_id"] == model_id for m in client.get("/api/v1/automl/models", headers=auth).json())
    assert client.get(f"/api/v1/automl/models/{model_id}/download", headers=auth).status_code == 200
    assert client.post("/api/v1/automl/train", files=files, data={"target": "nao_existe"}, headers=auth).status_code == 400


def test_background_jobs(client, auth):
    files = {"file": ("d.csv", _csv(), "text/csv")}
    job = client.post("/api/v1/automl/train", files=files, data={"target": "classe", "background": "true"}, headers=auth).json()["job_id"]
    evo = client.post(
        "/api/v1/automl/evolve", files=files, data={"target": "classe", "generations": 1, "population": 4}, headers=auth
    ).json()["job_id"]
    for jid in (job, evo):
        for _ in range(240):
            status = client.get(f"/api/v1/automl/jobs/{jid}", headers=auth).json()
            if status["status"] in ("SUCCESS", "FAILURE"):
                break
            time.sleep(0.25)
        assert status["status"] == "SUCCESS", status


def test_incremental_endpoint(client, auth):
    files = {"file": ("d.csv", _csv(), "text/csv")}
    r = client.post(
        "/api/v1/automl/incremental",
        files=files,
        data={"target": "classe", "problem_type": "classification", "batch_size": 16},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["accuracy"] > 0.8


def test_model_isolation(client, auth):
    from tests.conftest import register

    files = {"file": ("d.csv", _csv(), "text/csv")}
    model_id = client.post("/api/v1/automl/train", files=files, data={"target": "y"}, headers=auth).json()["model_id"]
    other = {"Authorization": f"Bearer {register(client)['token']}"}
    assert client.post("/api/v1/automl/predict", json={"model_id": model_id, "rows": [{"x1": 1}]}, headers=other).status_code == 404
