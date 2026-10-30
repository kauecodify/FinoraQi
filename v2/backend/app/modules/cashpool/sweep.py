from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from .compliance import ComplianceEngine
from .enums import MemberRole, PoolType, SweepDirection, SweepReason
from .forecast import LiquidityForecaster
from .fx import FXProvider
from .models import CashPool, FXConversion, PoolMember, SweepInstruction
from .money import d, q2
from .rails import PaymentRail

REASONS = {"zero": SweepReason.ZERO, "fixed": SweepReason.TARGET, "forecast": SweepReason.FORECAST}


class SweepEngine:
    def __init__(self, db: Session, fx: FXProvider, rail: PaymentRail) -> None:
        self.db, self.fx, self.rail = db, fx, rail

    def target(self, member: PoolMember, as_of: datetime) -> Decimal:
        rule = member.target_balance_rule or {"type": "zero"}
        kind = rule.get("type", "zero")
        if kind == "fixed":
            value = d(rule.get("amount", "0"))
        elif kind == "forecast":
            predicted = LiquidityForecaster(self.db).predict(member, int(rule.get("horizon_days", 7)), as_of)
            value = q2(max(predicted, Decimal("0")) * (1 + d(rule.get("buffer_pct", "0.10"))))
        else:
            value = Decimal("0")
        value = max(value, d(member.min_balance or 0))
        if member.max_balance is not None:
            value = min(value, d(member.max_balance))
        return value

    def plan(self, pool: CashPool, as_of: datetime) -> list[dict]:
        if pool.pool_type == PoolType.NOTIONAL:
            raise ValueError("pool nocional não movimenta saldos; use /positions para a posição consolidada")
        members = [m for m in pool.members if m.is_active]
        header = next((m for m in members if m.role == MemberRole.HEADER), None)
        if header is None:
            raise ValueError("pool sem conta header definida")
        min_amt = d(pool.sweep_policy.get("min_sweep_amount", "1.00"))
        compliance = ComplianceEngine(pool.compliance_profile)
        out = []
        for m in members:
            if m.role != MemberRole.PARTICIPANT:
                continue
            current = self.rail.get_balance(m, as_of)
            target = self.target(m, as_of)
            delta = q2(current - target)
            if abs(delta) < min_amt:
                continue
            if m.sweep_direction == SweepDirection.IN and delta > 0:
                continue
            if m.sweep_direction == SweepDirection.OUT and delta < 0:
                continue
            amount = abs(delta)
            amount_base = self.fx.convert(amount, m.currency, pool.base_currency)
            check = compliance.validate_sweep(amount_base, pool.base_currency)
            if not check.valid:
                raise ValueError("; ".join(check.errors))
            out.append(
                {
                    "member": m,
                    "header": header,
                    "direction": "to_header" if delta > 0 else "from_header",
                    "current": current,
                    "target": target,
                    "amount": amount,
                    "amount_header": q2(self.fx.convert(amount, m.currency, header.currency)),
                    "amount_base": q2(amount_base),
                    "reason": REASONS.get((m.target_balance_rule or {}).get("type", "zero"), SweepReason.MANUAL),
                    "flags": check.flags,
                }
            )
        return out

    def execute(self, pool: CashPool, as_of: datetime, dry_run: bool = False) -> list[dict]:
        plan = self.plan(pool, as_of)
        results = []
        for item in plan:
            m, header = item["member"], item["header"]
            to_header = item["direction"] == "to_header"
            instr = SweepInstruction(
                pool_id=pool.id,
                from_account_id=m.account_id if to_header else header.account_id,
                to_account_id=header.account_id if to_header else m.account_id,
                amount=item["amount"],
                currency=m.currency,
                reason=item["reason"],
                tax_flags=item["flags"],
                as_of=as_of,
                status="simulated" if dry_run else "pending",
            )
            if not dry_run:
                if to_header:
                    self.rail.transfer(m, header, item["amount"], item["amount_header"], as_of)
                else:
                    self.rail.transfer(header, m, item["amount_header"], item["amount"], as_of)
                if m.currency != header.currency:
                    frm, to = (m.currency, header.currency) if to_header else (header.currency, m.currency)
                    amt_in, amt_out = (item["amount"], item["amount_header"]) if to_header else (item["amount_header"], item["amount"])
                    self.db.add(
                        FXConversion(
                            pool_id=pool.id,
                            from_currency=frm,
                            to_currency=to,
                            amount_in=amt_in,
                            amount_out=amt_out,
                            rate=self.fx.rate(frm, to),
                            provider=self.fx.name,
                        )
                    )
                instr.status = "executed"
                instr.executed_at = as_of
                self.db.add(instr)
            results.append(
                {
                    "from_account_id": instr.from_account_id,
                    "to_account_id": instr.to_account_id,
                    "amount": str(q2(instr.amount)),
                    "currency": instr.currency,
                    "amount_base": str(item["amount_base"]),
                    "current_balance": str(q2(item["current"])),
                    "target_balance": str(q2(item["target"])),
                    "reason": instr.reason.value,
                    "status": instr.status,
                    "tax_flags": item["flags"],
                }
            )
        if not dry_run:
            self.db.commit()
        return results
