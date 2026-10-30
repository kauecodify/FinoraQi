from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings

from .compliance import ComplianceEngine
from .enums import NettingMode
from .forecast import LiquidityForecaster
from .fx import FXProvider, get_fx_provider
from .models import CashPool, CashPosition, IntercompanyObligation, NettingCycle, PoolMember, SweepInstruction, YieldAccrual
from .money import q2
from .netting import NettingEngine, member_by_account
from .rails import LedgerRail, PaymentRail
from .schemas import ObligationsImport, PoolCreate, PositionsImport
from .sweep import SweepEngine
from .yield_alloc import YieldAllocator


class NotFound(LookupError):
    pass


class CashPoolService:
    def __init__(self, db: Session, tenant_id: str, fx: FXProvider | None = None, rail: PaymentRail | None = None) -> None:
        self.db = db
        self.tenant_id = tenant_id
        self.fx = fx or get_fx_provider(get_settings().fx_provider)
        self.rail = rail or LedgerRail(db)

    def create_pool(self, payload: PoolCreate) -> CashPool:
        check = ComplianceEngine(payload.compliance_profile.value).validate_pool_creation(payload.members, payload.base_currency)
        if not check.valid:
            raise ValueError("; ".join(check.errors))
        for m in payload.members:
            self.fx.rate(m.currency, payload.base_currency)
        pool = CashPool(
            tenant_id=self.tenant_id,
            name=payload.name,
            pool_type=payload.pool_type,
            base_currency=payload.base_currency,
            compliance_profile=payload.compliance_profile.value,
            sweep_policy=payload.sweep_policy,
            netting_policy=payload.netting_policy,
            yield_policy=payload.yield_policy,
            members=[PoolMember(**m.model_dump()) for m in payload.members],
        )
        self.db.add(pool)
        self.db.commit()
        return self.get_pool(pool.id)

    def list_pools(self) -> list[CashPool]:
        stmt = select(CashPool).options(selectinload(CashPool.members)).where(CashPool.tenant_id == self.tenant_id).order_by(CashPool.id)
        return list(self.db.scalars(stmt))

    def get_pool(self, pool_id: int) -> CashPool:
        stmt = select(CashPool).options(selectinload(CashPool.members)).where(CashPool.id == pool_id, CashPool.tenant_id == self.tenant_id)
        pool = self.db.scalars(stmt).first()
        if pool is None or not pool.is_active:
            raise NotFound("pool não encontrado")
        return pool

    def import_positions(self, pool_id: int, payload: PositionsImport) -> int:
        pool = self.get_pool(pool_id)
        for p in payload.positions:
            m = member_by_account(pool, p.account_id)
            as_of = p.as_of.replace(tzinfo=None)
            self.db.add(CashPosition(pool_id=pool.id, member_id=m.id, as_of=as_of, balance=p.balance, currency=m.currency, source="import"))
        self.db.commit()
        return len(payload.positions)

    def positions(self, pool_id: int, as_of: datetime) -> dict:
        pool = self.get_pool(pool_id)
        as_of = as_of.replace(tzinfo=None)
        out, total = [], Decimal("0")
        for m in pool.members:
            if not m.is_active:
                continue
            bal = self.rail.get_balance(m, as_of)
            base = q2(self.fx.convert(bal, m.currency, pool.base_currency))
            total += base
            out.append(
                {
                    "member_id": m.id,
                    "entity_id": m.entity_id,
                    "account_id": m.account_id,
                    "role": m.role,
                    "currency": m.currency,
                    "balance": q2(bal),
                    "balance_base": base,
                    "base_currency": pool.base_currency,
                }
            )
        return {"as_of": as_of, "base_currency": pool.base_currency, "total_base": q2(total), "positions": out}

    def run_sweep(self, pool_id: int, as_of: datetime, dry_run: bool) -> dict:
        pool = self.get_pool(pool_id)
        instructions = SweepEngine(self.db, self.fx, self.rail).execute(pool, as_of.replace(tzinfo=None), dry_run)
        return {"pool_id": pool.id, "dry_run": dry_run, "count": len(instructions), "instructions": instructions}

    def add_obligations(self, pool_id: int, payload: ObligationsImport) -> int:
        pool = self.get_pool(pool_id)
        for o in payload.obligations:
            payer = member_by_account(pool, o.payer_account_id)
            payee = member_by_account(pool, o.payee_account_id)
            if payer.id == payee.id:
                raise ValueError("pagador e recebedor devem ser diferentes")
            self.fx.rate(o.currency, pool.base_currency)
            self.db.add(
                IntercompanyObligation(
                    pool_id=pool.id,
                    payer_member_id=payer.id,
                    payee_member_id=payee.id,
                    amount=o.amount,
                    currency=o.currency,
                    due_date=o.due_date,
                    reference=o.reference,
                )
            )
        self.db.commit()
        return len(payload.obligations)

    def run_netting(self, pool_id: int, cycle_date: date, mode: NettingMode | None, dry_run: bool) -> dict:
        pool = self.get_pool(pool_id)
        resolved = mode or NettingMode(pool.netting_policy.get("mode", "none"))
        return NettingEngine(self.db, self.fx, self.rail).run(pool, cycle_date, resolved, dry_run)

    def allocate_yield(self, pool_id: int, start: date, end: date, pool_yield: Decimal, dry_run: bool) -> dict:
        pool = self.get_pool(pool_id)
        rows = YieldAllocator(self.db, self.fx).allocate(pool, start, end, pool_yield, persist=not dry_run)
        return {"pool_id": pool.id, "method": pool.yield_policy.get("method", "pro_rata"), "dry_run": dry_run, "allocations": rows}

    def forecast(self, pool_id: int, member_id: int, horizon: int) -> dict:
        pool = self.get_pool(pool_id)
        member = next((m for m in pool.members if m.id == member_id), None)
        if member is None:
            raise NotFound("membro não encontrado")
        return {"member_id": member.id, "currency": member.currency, **LiquidityForecaster(self.db).forecast(member, horizon)}

    def history(self, pool_id: int, limit: int = 50) -> dict:
        pool = self.get_pool(pool_id)
        sweeps = self.db.scalars(
            select(SweepInstruction).where(SweepInstruction.pool_id == pool.id).order_by(SweepInstruction.id.desc()).limit(limit)
        ).all()
        cycles = self.db.scalars(
            select(NettingCycle).where(NettingCycle.pool_id == pool.id).order_by(NettingCycle.id.desc()).limit(limit)
        ).all()
        yields = self.db.scalars(
            select(YieldAccrual).where(YieldAccrual.pool_id == pool.id).order_by(YieldAccrual.id.desc()).limit(limit)
        ).all()
        return {
            "sweeps": [
                {
                    "id": s.id,
                    "from": s.from_account_id,
                    "to": s.to_account_id,
                    "amount": str(q2(s.amount)),
                    "currency": s.currency,
                    "reason": s.reason.value,
                    "status": s.status,
                    "tax_flags": s.tax_flags,
                    "as_of": s.as_of.isoformat(),
                }
                for s in sweeps
            ],
            "netting_cycles": [
                {
                    "id": c.id,
                    "cycle_date": c.cycle_date.isoformat(),
                    "mode": c.mode.value,
                    "gross_volume": str(q2(c.gross_volume)),
                    "net_volume": str(q2(c.net_volume)),
                    "savings_pct": str(c.savings_pct),
                }
                for c in cycles
            ],
            "yield_accruals": [
                {
                    "id": y.id,
                    "member_id": y.member_id,
                    "period_start": y.period_start.isoformat(),
                    "period_end": y.period_end.isoformat(),
                    "accrued": str(q2(y.accrued)),
                    "currency": y.currency,
                }
                for y in yields
            ],
        }
