from datetime import date
from decimal import Decimal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.core.deps import CurrentUser, DbSession
from app.db.models import CreditProfile
from app.ml.credit_risk import credit_score, default_probability, score_band

router = APIRouter(tags=["score"])


def normalize_cpf(value: str) -> str:
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) != 11 or digits == digits[0] * 11:
        raise ValueError("CPF inválido")
    for size in (9, 10):
        total = sum(int(digits[i]) * (size + 1 - i) for i in range(size))
        check = (total * 10) % 11 % 10
        if check != int(digits[size]):
            raise ValueError("CPF inválido")
    return digits


class CreditOperationIn(BaseModel):
    cpf: str
    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    paid_on_time: bool
    paid_at: date | None = None
    income_ratio: float = Field(default=0.3, ge=0, le=1)

    @field_validator("cpf")
    @classmethod
    def _cpf(cls, v: str) -> str:
        return normalize_cpf(v)


def _view(p: CreditProfile, income_ratio: float = 0.3) -> dict:
    pd_ = default_probability(p.on_time_payments, p.late_payments, float(p.total_paid), len(p.history or []), income_ratio)
    return {
        "cpf": f"***.{p.cpf[3:6]}.{p.cpf[6:9]}-**",
        "score": p.score,
        "band": score_band(p.score),
        "probability_of_default": round(pd_, 4),
        "on_time_payments": p.on_time_payments,
        "late_payments": p.late_payments,
        "total_paid": str(p.total_paid),
        "operations": len(p.history or []),
    }


@router.post("/credit-operation")
def register_operation(payload: CreditOperationIn, user: CurrentUser, db: DbSession):
    profile = db.scalar(select(CreditProfile).where(CreditProfile.tenant_id == user.tenant_id, CreditProfile.cpf == payload.cpf))
    if profile is None:
        profile = CreditProfile(
            tenant_id=user.tenant_id, cpf=payload.cpf, score=500, on_time_payments=0, late_payments=0, total_paid=Decimal("0"), history=[]
        )
        db.add(profile)
    if payload.paid_on_time:
        profile.on_time_payments += 1
    else:
        profile.late_payments += 1
    profile.total_paid = Decimal(profile.total_paid or 0) + payload.amount
    profile.history = [
        *(profile.history or []),
        {"amount": str(payload.amount), "on_time": payload.paid_on_time, "date": (payload.paid_at or date.today()).isoformat()},
    ][-500:]
    pd_ = default_probability(
        profile.on_time_payments, profile.late_payments, float(profile.total_paid), len(profile.history), payload.income_ratio
    )
    profile.score = credit_score(pd_)
    db.commit()
    return _view(profile, payload.income_ratio)


@router.get("/score/{cpf}")
def get_score(cpf: str, user: CurrentUser, db: DbSession):
    try:
        digits = normalize_cpf(cpf)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    profile = db.scalar(select(CreditProfile).where(CreditProfile.tenant_id == user.tenant_id, CreditProfile.cpf == digits))
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "CPF sem histórico")
    return _view(profile)
