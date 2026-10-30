from decimal import Decimal

from app.modules.cashpool.compliance import ComplianceEngine
from app.modules.cashpool.enums import NettingMode
from app.modules.cashpool.fx import StaticFXProvider
from app.modules.cashpool.netting import compute, net_multilateral
from tests.conftest import register

API = "/api/v1/cashpool"


def _pool(client, auth, **extra):
    payload = {
        "name": "Grupo Alfa",
        "pool_type": "physical",
        "base_currency": "BRL",
        "compliance_profile": "BR",
        "netting_policy": {"mode": "multilateral"},
        "members": [
            {"entity_id": "holding", "account_id": "ACC-H", "currency": "BRL", "role": "header"},
            {"entity_id": "sub-a", "account_id": "ACC-A", "currency": "BRL"},
            {"entity_id": "sub-b", "account_id": "ACC-B", "currency": "BRL", "min_balance": "1000"},
            {"entity_id": "sub-c", "account_id": "ACC-C", "currency": "USD", "target_balance_rule": {"type": "fixed", "amount": "500"}},
        ],
        **extra,
    }
    r = client.post(f"{API}/pools", json=payload, headers=auth)
    assert r.status_code == 201, r.text
    return r.json()


def _positions(client, auth, pool_id, items, as_of="2026-03-01T10:00:00"):
    r = client.post(
        f"{API}/pools/{pool_id}/positions",
        json={"positions": [{"account_id": a, "balance": b, "as_of": as_of} for a, b in items]},
        headers=auth,
    )
    assert r.status_code == 201, r.text


def test_netting_algorithm_reduces_transfers():
    obligations = [(1, 2, Decimal("100000")), (2, 3, Decimal("120000")), (3, 1, Decimal("110000")), (2, 1, Decimal("20000"))]
    result = compute(obligations, NettingMode.MULTILATERAL)
    assert result["gross"] == Decimal("350000.00")
    assert len(result["settlements"]) <= 2
    assert result["net"] < result["gross"]
    pos = {}
    for p, r, a in net_multilateral(obligations):
        pos[p] = pos.get(p, 0) - a
        pos[r] = pos.get(r, 0) + a
    assert pos == {1: Decimal("30000"), 2: Decimal("-40000"), 3: Decimal("10000")}
    assert result["savings_pct"] == Decimal("88.57")


def test_compliance_rules():
    eng = ComplianceEngine("BR")
    assert "iof_0.38pct" in eng.validate_sweep(Decimal("50000"), "BRL").flags
    assert ComplianceEngine("ZZ").validate_sweep(Decimal("1"), "USD").valid
    assert not eng.validate_sweep(Decimal("0"), "BRL").valid


def test_fx_static():
    fx = StaticFXProvider()
    assert fx.convert(Decimal("100"), "USD", "BRL") == Decimal("540.0000")
    assert fx.rate("BRL", "BRL") == 1


def test_create_pool_validation(client, auth):
    bad = {"name": "X1", "base_currency": "BRL", "members": [{"entity_id": "a", "account_id": "A", "currency": "BRL"}]}
    r = client.post(f"{API}/pools", json=bad, headers=auth)
    assert r.status_code == 400 and "header" in r.text
    bad["base_currency"] = "brl"
    assert client.post(f"{API}/pools", json=bad, headers=auth).status_code == 422


def test_sweep_moves_balances_to_header(client, auth):
    pool = _pool(client, auth)
    pid = pool["id"]
    _positions(client, auth, pid, [("ACC-H", "0"), ("ACC-A", "50000"), ("ACC-B", "-2000"), ("ACC-C", "1500")])
    dry = client.post(f"{API}/pools/{pid}/sweep?dry_run=true&as_of=2026-03-01T12:00:00", headers=auth).json()
    assert dry["count"] == 3 and all(i["status"] == "simulated" for i in dry["instructions"])
    a = next(i for i in dry["instructions"] if i["from_account_id"] == "ACC-A")
    assert a["amount"] == "50000.00" and "iof_0.38pct" in a["tax_flags"]
    run = client.post(f"{API}/pools/{pid}/sweep?as_of=2026-03-01T12:00:00", headers=auth).json()
    assert run["count"] == 3
    pos = client.get(f"{API}/pools/{pid}/positions?as_of=2026-03-01T13:00:00", headers=auth).json()
    bal = {p["account_id"]: Decimal(p["balance"]) for p in pos["positions"]}
    assert bal["ACC-A"] == 0 and bal["ACC-B"] == Decimal("1000.00") and bal["ACC-C"] == Decimal("500.00")
    assert bal["ACC-H"] == Decimal("50000") - Decimal("3000") + Decimal("5400")
    assert Decimal(pos["total_base"]) == Decimal("50000") - Decimal("2000") + Decimal("8100")
    again = client.post(f"{API}/pools/{pid}/sweep?as_of=2026-03-01T14:00:00", headers=auth).json()
    assert again["count"] == 0
    hist = client.get(f"{API}/pools/{pid}/history", headers=auth).json()
    assert len(hist["sweeps"]) == 3


def test_netting_endpoint(client, auth):
    pid = _pool(client, auth)["id"]
    _positions(client, auth, pid, [("ACC-H", "100000"), ("ACC-A", "100000"), ("ACC-B", "100000"), ("ACC-C", "0")])
    obligations = [
        {"payer_account_id": "ACC-H", "payee_account_id": "ACC-A", "amount": "100000", "currency": "BRL", "due_date": "2026-03-10"},
        {"payer_account_id": "ACC-A", "payee_account_id": "ACC-B", "amount": "120000", "currency": "BRL", "due_date": "2026-03-10"},
        {"payer_account_id": "ACC-B", "payee_account_id": "ACC-H", "amount": "110000", "currency": "BRL", "due_date": "2026-03-10"},
        {"payer_account_id": "ACC-A", "payee_account_id": "ACC-H", "amount": "20000", "currency": "BRL", "due_date": "2026-03-10"},
        {"payer_account_id": "ACC-A", "payee_account_id": "ACC-B", "amount": "999", "currency": "BRL", "due_date": "2026-04-30"},
    ]
    assert client.post(f"{API}/pools/{pid}/obligations", json={"obligations": obligations}, headers=auth).status_code == 201
    dry = client.post(f"{API}/pools/{pid}/netting", json={"cycle_date": "2026-03-10", "dry_run": True}, headers=auth).json()
    assert dry["obligations"] == 4 and dry["gross_volume"] == "350000.00" and dry["transfers_net"] <= 2
    run = client.post(f"{API}/pools/{pid}/netting", json={"cycle_date": "2026-03-10"}, headers=auth).json()
    assert run["cycle_id"] is not None and Decimal(run["savings_pct"]) > 85
    again = client.post(f"{API}/pools/{pid}/netting", json={"cycle_date": "2026-03-10"}, headers=auth).json()
    assert again["obligations"] == 0
    pos = client.get(f"{API}/pools/{pid}/positions", headers=auth).json()
    bal = {p["account_id"]: Decimal(p["balance"]) for p in pos["positions"]}
    assert bal["ACC-H"] == Decimal("130000") and bal["ACC-A"] == Decimal("60000") and bal["ACC-B"] == Decimal("110000")
    none = client.post(f"{API}/pools/{pid}/netting", json={"cycle_date": "2026-03-10", "mode": "none"}, headers=auth)
    assert none.status_code == 400


def test_yield_allocation_sums_exactly(client, auth):
    pid = _pool(client, auth, yield_policy={"method": "tiered", "header_spread_bps": 50})["id"]
    _positions(
        client, auth, pid, [("ACC-H", "100000"), ("ACC-A", "50000"), ("ACC-B", "25000"), ("ACC-C", "0")], as_of="2026-02-01T00:00:00"
    )
    _positions(client, auth, pid, [("ACC-A", "0")], as_of="2026-02-15T00:00:00")
    r = client.post(
        f"{API}/pools/{pid}/yield", json={"period_start": "2026-02-01", "period_end": "2026-02-28", "pool_yield": "1234.57"}, headers=auth
    )
    assert r.status_code == 200, r.text
    alloc = {a["account_id"]: a for a in r.json()["allocations"]}
    assert sum(Decimal(a["accrued"]) for a in alloc.values()) == Decimal("1234.57")
    assert Decimal(alloc["ACC-A"]["twab"]) == Decimal("25000.00")
    assert Decimal(alloc["ACC-C"]["accrued"]) == 0
    assert Decimal(alloc["ACC-H"]["rate"]) > Decimal(alloc["ACC-B"]["rate"])


def test_forecast_target_and_endpoint(client, auth):
    pool = _pool(client, auth)
    pid = pool["id"]
    member_b = next(m for m in pool["members"] if m["account_id"] == "ACC-B")
    items = []
    for day in range(1, 29):
        items.append({"account_id": "ACC-B", "balance": str(1000 + day * 100), "as_of": f"2026-02-{day:02d}T18:00:00"})
    client.post(f"{API}/pools/{pid}/positions", json={"positions": items}, headers=auth)
    r = client.get(f"{API}/pools/{pid}/forecast/{member_b['id']}?horizon=5", headers=auth).json()
    assert r["observations"] == 28 and len(r["p50"]) == 5 and r["p50"][-1] > 3700


def test_tenant_isolation_and_flags(client, auth):
    pid = _pool(client, auth)["id"]
    other = {"Authorization": f"Bearer {register(client)['token']}"}
    assert client.get(f"{API}/pools/{pid}", headers=other).status_code == 404
    assert client.get(f"{API}/pools", headers=other).json() == []
    assert client.put("/api/v1/flags/cashpool.sweep", json={"enabled": False}, headers=auth).status_code == 200
    assert client.post(f"{API}/pools/{pid}/sweep", headers=auth).status_code == 403
    client.put("/api/v1/flags/cashpool.enabled", json={"enabled": False}, headers=auth)
    assert client.get(f"{API}/pools", headers=auth).status_code == 403
    assert client.get(f"{API}/pools", headers=other).status_code == 200


def test_notional_pool_rejects_sweep(client, auth):
    pid = _pool(client, auth, pool_type="notional")["id"]
    r = client.post(f"{API}/pools/{pid}/sweep", headers=auth)
    assert r.status_code == 400 and "nocional" in r.text


def test_landing_page_netting_example_matches_engine():
    br, mx, us = 1, 2, 3
    obligations = [(br, mx, Decimal("120000")), (mx, us, Decimal("90000")), (us, br, Decimal("100000")), (mx, br, Decimal("40000"))]
    result = compute(obligations, NettingMode.MULTILATERAL)
    assert result["gross"] == Decimal("350000.00") and result["net"] == Decimal("20000.00")
    assert len(result["settlements"]) == 2 and result["savings_pct"] == Decimal("94.29")
    assert {(p, r) for p, r, _ in result["settlements"]} == {(mx, br), (us, br)}
