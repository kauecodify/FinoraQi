from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .enums import MemberRole, YieldMethod
from .fx import FXProvider
from .models import CashPool, CashPosition, PoolMember, YieldAccrual
from .money import d, q2, q4


class YieldAllocator:
    def __init__(self, db: Session, fx: FXProvider) -> None:
        self.db, self.fx = db, fx

    def twab(self, member: PoolMember, start: date, end: date, currency: str) -> Decimal:
        """Saldo médio ponderado no tempo (somente saldos positivos), na moeda informada."""
        rows = self.db.scalars(
            select(CashPosition)
            .where(CashPosition.member_id == member.id, CashPosition.as_of <= datetime.combine(end, time.max))
            .order_by(CashPosition.as_of, CashPosition.id)
        ).all()
        days = (end - start).days + 1
        idx, balance, total = 0, Decimal("0"), Decimal("0")
        for i in range(days):
            day_end = datetime.combine(start + timedelta(days=i), time.max)
            while idx < len(rows) and rows[idx].as_of <= day_end:
                balance = d(rows[idx].balance)
                idx += 1
            total += max(balance, Decimal("0"))
        avg = total / days
        return q4(self.fx.convert(avg, member.currency, currency))

    def allocate(self, pool: CashPool, start: date, end: date, pool_yield: Decimal, persist: bool = True) -> list[dict]:
        if end < start:
            raise ValueError("period_end deve ser maior ou igual a period_start")
        members = [m for m in pool.members if m.is_active]
        twabs = {m.id: self.twab(m, start, end, pool.base_currency) for m in members}
        total = sum(twabs.values(), Decimal("0"))
        if total <= 0:
            raise ValueError("saldo médio do pool é zero no período; importe posições antes")
        policy = pool.yield_policy or {}
        method = YieldMethod(policy.get("method", YieldMethod.PRO_RATA.value))
        days = (end - start).days + 1
        accrued: dict[int, Decimal] = {}
        if method == YieldMethod.CONTRACT:
            rates = policy.get("contract_rates", {})
            for m in members:
                annual = d(rates.get(m.account_id, rates.get(str(m.id), "0")))
                accrued[m.id] = q2(twabs[m.id] * annual * days / 365)
        else:
            spread = d(policy.get("header_spread_bps", 25)) / Decimal("10000") if method == YieldMethod.TIERED else Decimal("0")
            weights = {m.id: twabs[m.id] * (1 + spread if m.role == MemberRole.HEADER else 1 - spread) for m in members}
            wsum = sum(weights.values(), Decimal("0"))
            running = Decimal("0")
            ordered = [m for m in members if weights[m.id] > 0]
            for i, m in enumerate(ordered):
                share = q2(d(pool_yield) * weights[m.id] / wsum) if i < len(ordered) - 1 else q2(d(pool_yield) - running)
                accrued[m.id] = share
                running += share
            for m in members:
                accrued.setdefault(m.id, Decimal("0.00"))
        out = []
        for m in members:
            base = twabs[m.id]
            rate = (accrued[m.id] / base).quantize(Decimal("0.00000001")) if base > 0 else Decimal("0")
            if persist:
                self.db.add(
                    YieldAccrual(
                        pool_id=pool.id,
                        member_id=m.id,
                        period_start=start,
                        period_end=end,
                        base_amount=base,
                        rate=rate,
                        accrued=accrued[m.id],
                        currency=pool.base_currency,
                    )
                )
            out.append(
                {
                    "member_id": m.id,
                    "account_id": m.account_id,
                    "twab": str(q2(base)),
                    "rate": str(rate),
                    "accrued": str(accrued[m.id]),
                    "currency": pool.base_currency,
                }
            )
        if persist:
            self.db.commit()
        return out
