import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum

import jwt
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.deps import CurrentUser, DbSession
from app.core.security import decode_access_token
from app.core.websocket import manager
from app.db.models import PaymentRecord, utcnow
from app.ml import scoring
from app.ml.anomaly import detect_anomaly

router = APIRouter(tags=["payments"])


class Category(str, Enum):
    payment = "payment"
    transfer = "transfer"
    withdrawal = "withdrawal"
    deposit = "deposit"
    refund = "refund"


class Method(str, Enum):
    pix = "pix"
    debit = "debit"
    credit = "credit"
    boleto = "boleto"
    crypto = "crypto"


class PaymentIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_\-.@]+$")
    amount: Decimal = Field(gt=0, le=Decimal("10000000"), max_digits=12, decimal_places=2)
    description: str = Field(min_length=1, max_length=200)
    category: Category = Category.payment
    payment_method: Method = Method.pix
    timestamp: datetime | None = None
    user_trust: float = Field(default=0.5, ge=0, le=1)


class PaymentOut(BaseModel):
    transaction_id: str
    user_id: str
    amount: Decimal
    status: str
    risk_score: float
    risk_level: str
    is_anomaly: bool
    reasons: list[str]
    processing_time_ms: float
    created_at: datetime


class BatchIn(BaseModel):
    payments: list[PaymentIn] = Field(min_length=1, max_length=1000)


def _history(db, tenant: str, user_ref: str, limit: int = 50) -> list[float]:
    rows = db.scalars(
        select(PaymentRecord.amount)
        .where(PaymentRecord.tenant_id == tenant, PaymentRecord.user_ref == user_ref)
        .order_by(PaymentRecord.id.desc())
        .limit(limit)
    ).all()
    return [float(a) for a in rows]


def _features(p: PaymentIn, history: list[float]) -> scoring.PaymentFeatures:
    return scoring.PaymentFeatures(
        amount=float(p.amount),
        payment_method=p.payment_method.value,
        category=p.category.value,
        timestamp=p.timestamp or utcnow(),
        recent_amounts=tuple(history),
        user_trust=p.user_trust,
    )


def _persist(db, tenant: str, p: PaymentIn, score: float, anomaly: bool, reasons: list[str], elapsed: float) -> PaymentOut:
    level = scoring.risk_level(score)
    rec = PaymentRecord(
        transaction_id=f"txn_{uuid.uuid4().hex[:20]}",
        tenant_id=tenant,
        user_ref=p.user_id,
        amount=p.amount,
        description=p.description,
        category=p.category.value,
        payment_method=p.payment_method.value,
        risk_score=round(score, 4),
        risk_level=level,
        status=scoring.decide(score, anomaly),
        is_anomaly=anomaly,
        processing_time_ms=round(elapsed, 3),
    )
    db.add(rec)
    return PaymentOut(
        transaction_id=rec.transaction_id,
        user_id=p.user_id,
        amount=p.amount,
        status=rec.status,
        risk_score=rec.risk_score,
        risk_level=level,
        is_anomaly=anomaly,
        reasons=reasons,
        processing_time_ms=rec.processing_time_ms,
        created_at=utcnow(),
    )


@router.post("/payments/process", response_model=PaymentOut)
async def process_payment(payload: PaymentIn, user: CurrentUser, db: DbSession):
    start = time.perf_counter()
    history = _history(db, user.tenant_id, payload.user_id)
    score = scoring.score_payment(_features(payload, history))
    anomaly, reasons = detect_anomaly(float(payload.amount), history, payload.payment_method.value)
    out = _persist(db, user.tenant_id, payload, score, anomaly, reasons, (time.perf_counter() - start) * 1000)
    db.commit()
    await manager.send(f"payments:{user.tenant_id}:{payload.user_id}", {"type": "payment_processed", "data": out.model_dump(mode="json")})
    return out


@router.post("/payments/batch")
def process_batch(payload: BatchIn, user: CurrentUser, db: DbSession):
    start = time.perf_counter()
    histories: dict[str, list[float]] = {}
    vectors, anomalies = [], []
    for p in payload.payments:
        hist = histories.setdefault(p.user_id, _history(db, user.tenant_id, p.user_id))
        vectors.append(_features(p, hist).vector())
        anomalies.append(detect_anomaly(float(p.amount), hist, p.payment_method.value))
        hist.insert(0, float(p.amount))
    scores = scoring.risk_scores(np.vstack(vectors))
    elapsed = (time.perf_counter() - start) * 1000
    per_item = elapsed / len(payload.payments)
    results = [
        _persist(db, user.tenant_id, p, float(s), a[0], a[1], per_item)
        for p, s, a in zip(payload.payments, scores, anomalies, strict=False)
    ]
    db.commit()
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "total": len(results),
        "by_status": counts,
        "processing_time_ms": round(elapsed, 3),
        "results": [r.model_dump(mode="json") for r in results],
    }


@router.get("/payments")
def list_payments(user: CurrentUser, db: DbSession, limit: int = Query(100, ge=1, le=1000)):
    rows = db.scalars(
        select(PaymentRecord).where(PaymentRecord.tenant_id == user.tenant_id).order_by(PaymentRecord.id.desc()).limit(limit)
    ).all()
    return [
        {
            "transaction_id": r.transaction_id,
            "user_id": r.user_ref,
            "amount": str(r.amount),
            "status": r.status,
            "risk_score": r.risk_score,
            "risk_level": r.risk_level,
            "is_anomaly": r.is_anomaly,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.get("/payments/benchmark")
def benchmark(user: CurrentUser, n: int = Query(10_000, ge=100, le=200_000)):
    rng = np.random.default_rng(7)
    X = rng.random((n, len(scoring.FEATURE_NAMES)))
    scoring.risk_scores(X[:10], use_numba=True)
    t0 = time.perf_counter()
    a = scoring.risk_scores(X, use_numba=True)
    t1 = time.perf_counter()
    b = scoring.risk_scores(X, use_numba=False)
    t2 = time.perf_counter()
    return {
        "n": n,
        "numba_available": scoring.NUMBA_AVAILABLE,
        "numba_ms": round((t1 - t0) * 1000, 3),
        "numpy_ms": round((t2 - t1) * 1000, 3),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "throughput_per_s": int(n / max(t1 - t0, 1e-9)),
    }


@router.websocket("/ws/payments/{user_id}")
async def payments_ws(websocket: WebSocket, user_id: str, token: str = ""):
    try:
        tenant = decode_access_token(token)["tenant"]
    except (jwt.PyJWTError, KeyError):
        await websocket.close(code=4401)
        return
    channel = f"payments:{tenant}:{user_id}"
    await manager.connect(websocket, channel)
    try:
        await websocket.send_json({"type": "connected", "channel": user_id})
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, channel)
