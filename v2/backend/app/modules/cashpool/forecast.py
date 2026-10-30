from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import utcnow
from app.ml.forecast import CashFlowModel

from .models import CashPosition, PoolMember
from .money import q2


class LiquidityForecaster:
    """Projeta o saldo de um membro a partir do histórico diário em cash_positions."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def history(self, member: PoolMember, days: int = 90, as_of: datetime | None = None) -> tuple[list, list[float]]:
        if as_of is None:
            latest = self.db.scalar(select(func.max(CashPosition.as_of)).where(CashPosition.member_id == member.id))
            as_of = latest or utcnow()
        end = as_of
        rows = self.db.scalars(
            select(CashPosition)
            .where(CashPosition.member_id == member.id, CashPosition.as_of >= end - timedelta(days=days), CashPosition.as_of <= end)
            .order_by(CashPosition.as_of, CashPosition.id)
        ).all()
        daily: dict = {}
        for r in rows:
            daily[r.as_of.date()] = float(r.balance)
        dates = sorted(daily)
        return dates, [daily[d] for d in dates]

    def forecast(self, member: PoolMember, horizon_days: int = 7, as_of: datetime | None = None) -> dict:
        dates, values = self.history(member, as_of=as_of)
        if not dates:
            zero = [0.0] * horizon_days
            return {"dates": [], "p10": zero, "p50": zero, "p90": zero, "observations": 0}
        out = CashFlowModel().fit(dates, values).predict(horizon_days, last_date=dates[-1])
        out["observations"] = len(dates)
        return out

    def predict(self, member: PoolMember, horizon_days: int = 7, as_of: datetime | None = None) -> Decimal:
        return q2(self.forecast(member, horizon_days, as_of)["p50"][-1])
