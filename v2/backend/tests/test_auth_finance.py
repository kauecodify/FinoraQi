from tests.conftest import register


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/api/v1/health").status_code == 200


def test_register_login_profile(client):
    data = register(client, "Ana")
    email = data["user"]["email"]
    assert client.post("/api/v1/auth/register", json={"nome": "Xavier", "email": email, "senha": "outra-senha-123"}).status_code == 409
    assert client.post("/api/v1/auth/login", json={"email": email, "senha": "errada-123"}).status_code == 401
    r = client.post("/api/v1/auth/login", json={"email": email, "senha": "senha-segura-123"})
    assert r.status_code == 200
    h = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.put("/api/v1/profile", json={"telefone": "11999990000"}, headers=h).json()["telefone"] == "11999990000"


def test_requires_token(client):
    assert client.get("/api/v1/transactions").status_code == 401
    assert client.get("/api/v1/transactions", headers={"Authorization": "Bearer invalido"}).status_code == 401


def test_transactions_crud_and_ai_category(client, auth):
    r = client.post(
        "/api/v1/transactions",
        json={"type": "expense", "amount": "45.90", "description": "Uber viagem centro", "date": "2026-01-10"},
        headers=auth,
    )
    assert r.status_code == 201, r.text
    tx = r.json()
    assert tx["category"] == "transporte" and tx["category_source"] == "ai"
    r = client.put(
        f"/api/v1/transactions/{tx['id']}",
        json={"type": "expense", "amount": "50", "category": "lazer", "date": "2026-01-10"},
        headers=auth,
    )
    assert r.json()["category_source"] == "user"
    assert len(client.get("/api/v1/transactions", headers=auth).json()) == 1
    assert client.delete(f"/api/v1/transactions/{tx['id']}", headers=auth).status_code == 204
    assert client.get("/api/v1/transactions", headers=auth).json() == []


def test_isolation_between_users(client, auth):
    tx = client.post(
        "/api/v1/transactions", json={"type": "income", "amount": "100", "category": "salario", "date": "2026-01-01"}, headers=auth
    ).json()
    other = {"Authorization": f"Bearer {register(client)['token']}"}
    assert client.delete(f"/api/v1/transactions/{tx['id']}", headers=other).status_code == 404


def test_goals_wallets_investments_summary(client, auth):
    g = client.post("/api/v1/goals", json={"name": "Reserva", "target": "1000", "saved": "250"}, headers=auth).json()
    assert g["progress"] == 0.25
    client.post("/api/v1/wallets", json={"name": "Conta", "balance": "300"}, headers=auth)
    client.post("/api/v1/investments", json={"name": "CDB", "amount": "500", "rate": 12.5, "date": "2026-01-01"}, headers=auth)
    s = client.get("/api/v1/dashboard/summary", headers=auth).json()
    assert s["invested"] == "500.00" and s["wallets"] == "300.00"


def test_validation_errors(client, auth):
    assert client.post("/api/v1/transactions", json={"type": "other", "amount": "-1", "date": "x"}, headers=auth).status_code == 422
