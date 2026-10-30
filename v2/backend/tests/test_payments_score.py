VALID_CPF = "52998224725"


def test_process_payment_low_risk(client, auth):
    r = client.post(
        "/api/v1/payments/process",
        json={
            "user_id": "cli_1",
            "amount": "120.00",
            "description": "Compra",
            "payment_method": "pix",
            "timestamp": "2026-03-04T14:00:00",
            "user_trust": 0.9,
        },
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_level"] == "low" and body["status"] == "approved"
    assert 0 <= body["risk_score"] < 0.35


def test_high_risk_is_rejected_not_approved(client, auth):
    r = client.post(
        "/api/v1/payments/process",
        json={
            "user_id": "cli_2",
            "amount": "9000000",
            "description": "Saque",
            "category": "withdrawal",
            "payment_method": "crypto",
            "timestamp": "2026-03-07T03:00:00",
            "user_trust": 0.0,
        },
        headers=auth,
    )
    body = r.json()
    assert body["risk_level"] == "high" and body["status"] == "rejected"


def test_anomaly_detection_flags_review(client, auth):
    for amount in ["100", "110", "95", "105", "98", "102"]:
        client.post(
            "/api/v1/payments/process",
            json={"user_id": "cli_3", "amount": amount, "description": "x", "timestamp": "2026-03-04T14:00:00", "user_trust": 0.9},
            headers=auth,
        )
    body = client.post(
        "/api/v1/payments/process",
        json={"user_id": "cli_3", "amount": "5000", "description": "x", "timestamp": "2026-03-04T14:00:00", "user_trust": 0.9},
        headers=auth,
    ).json()
    assert body["is_anomaly"] is True and body["status"] in ("review", "rejected")


def test_batch_and_benchmark(client, auth):
    payments = [{"user_id": f"b{i % 3}", "amount": str(50 + i), "description": "lote"} for i in range(30)]
    r = client.post("/api/v1/payments/batch", json={"payments": payments}, headers=auth)
    assert r.status_code == 200 and r.json()["total"] == 30
    bench = client.get("/api/v1/payments/benchmark?n=1000", headers=auth).json()
    assert bench["max_abs_diff"] < 1e-9


def test_payment_validation(client, auth):
    assert (
        client.post("/api/v1/payments/process", json={"user_id": "a b", "amount": "0", "description": ""}, headers=auth).status_code == 422
    )


def test_websocket_requires_token(client, auth):
    token = auth["Authorization"].split()[1]
    with client.websocket_connect(f"/api/v1/ws/payments/cli_ws?token={token}") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_text("ping")
        assert ws.receive_json()["type"] == "pong"


def test_credit_score_flow(client, auth):
    assert client.get(f"/api/v1/score/{VALID_CPF}", headers=auth).status_code == 404
    assert (
        client.post("/api/v1/credit-operation", json={"cpf": "11111111111", "amount": "10", "paid_on_time": True}, headers=auth).status_code
        == 422
    )
    good = None
    for _ in range(6):
        good = client.post(
            "/api/v1/credit-operation", json={"cpf": "529.982.247-25", "amount": "1500", "paid_on_time": True}, headers=auth
        ).json()
    bad = client.post("/api/v1/credit-operation", json={"cpf": VALID_CPF, "amount": "100", "paid_on_time": False}, headers=auth).json()
    assert 300 <= bad["score"] <= 900
    assert bad["score"] < good["score"]
    assert client.get(f"/api/v1/score/{VALID_CPF}", headers=auth).json()["operations"] == 7
