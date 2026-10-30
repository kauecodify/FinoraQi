"""Netting bilateral e multilateral de obrigações intercompany.

A versão do anexo zerava os deltas do sweep sem registrar quem paga quem. Aqui o netting
opera sobre obrigações reais (contas a pagar/receber entre membros) e gera o conjunto
mínimo de liquidações; no modo multilateral o número de transferências é no máximo n-1.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import utcnow

from .enums import NettingMode
from .fx import FXProvider
from .models import CashPool, FXConversion, IntercompanyObligation, NettingCycle, PoolMember
from .money import q2, q4
from .rails import PaymentRail

Obligation = tuple[int, int, Decimal]


def net_bilateral(obligations: list[Obligation]) -> list[Obligation]:
    pair: dict[tuple[int, int], Decimal] = defaultdict(Decimal)
    for payer, payee, amount in obligations:
        a, b = sorted((payer, payee))
        pair[(a, b)] += amount if payer == a else -amount
    out = []
    for (a, b), value in sorted(pair.items()):
        value = q4(value)
        if value > 0:
            out.append((a, b, value))
        elif value < 0:
            out.append((b, a, -value))
    return out


def net_positions(obligations: list[Obligation]) -> dict[int, Decimal]:
    pos: dict[int, Decimal] = defaultdict(Decimal)
    for payer, payee, amount in obligations:
        pos[payer] -= amount
        pos[payee] += amount
    return dict(pos)


def net_multilateral(obligations: list[Obligation]) -> list[Obligation]:
    pos = net_positions(obligations)
    creditors = sorted(([m, v] for m, v in pos.items() if v > 0), key=lambda x: (-x[1], x[0]))
    debtors = sorted(([m, -v] for m, v in pos.items() if v < 0), key=lambda x: (-x[1], x[0]))
    out: list[Obligation] = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        amount = min(debtors[i][1], creditors[j][1])
        if q4(amount) > 0:
            out.append((debtors[i][0], creditors[j][0], q4(amount)))
        debtors[i][1] -= amount
        creditors[j][1] -= amount
        if debtors[i][1] <= Decimal("0.00005"):
            i += 1
        if creditors[j][1] <= Decimal("0.00005"):
            j += 1
    return out


def compute(obligations: list[Obligation], mode: NettingMode) -> dict:
    gross = sum((a for _, _, a in obligations), Decimal("0"))
    if mode == NettingMode.MULTILATERAL:
        settlements = net_multilateral(obligations)
    elif mode == NettingMode.BILATERAL:
        settlements = net_bilateral(obligations)
    else:
        settlements = list(obligations)
    net = sum((a for _, _, a in settlements), Decimal("0"))
    savings = (gross - net) / gross * 100 if gross else Decimal("0")
    return {"gross": q2(gross), "net": q2(net), "savings_pct": q2(savings), "settlements": settlements}


class NettingEngine:
    def __init__(self, db: Session, fx: FXProvider, rail: PaymentRail) -> None:
        self.db, self.fx, self.rail = db, fx, rail

    def run(self, pool: CashPool, cycle_date: date, mode: NettingMode, dry_run: bool = False) -> dict:
        if mode == NettingMode.NONE:
            raise ValueError("netting desabilitado para este pool (mode=none)")
        members = {m.id: m for m in pool.members if m.is_active}
        rows = self.db.scalars(
            select(IntercompanyObligation).where(
                IntercompanyObligation.pool_id == pool.id,
                IntercompanyObligation.status == "open",
                IntercompanyObligation.due_date <= cycle_date,
            )
        ).all()
        rows = [r for r in rows if r.payer_member_id in members and r.payee_member_id in members]
        base = pool.base_currency
        obligations = [(r.payer_member_id, r.payee_member_id, self.fx.convert(r.amount, r.currency, base)) for r in rows]
        result = compute(obligations, mode)
        settlements = [
            {
                "payer_member_id": p,
                "payer_account_id": members[p].account_id,
                "payee_member_id": r,
                "payee_account_id": members[r].account_id,
                "amount": str(q2(a)),
                "currency": base,
            }
            for p, r, a in result["settlements"]
        ]
        summary = {
            "cycle_date": cycle_date.isoformat(),
            "mode": mode.value,
            "obligations": len(rows),
            "transfers_gross": len(obligations),
            "transfers_net": len(settlements),
            "gross_volume": str(result["gross"]),
            "net_volume": str(result["net"]),
            "savings_pct": str(result["savings_pct"]),
            "currency": base,
            "settlements": settlements,
            "dry_run": dry_run,
            "cycle_id": None,
        }
        if dry_run or not rows:
            return summary

        cycle = NettingCycle(
            pool_id=pool.id,
            cycle_date=cycle_date,
            mode=mode,
            gross_volume=result["gross"],
            net_volume=result["net"],
            savings_pct=result["savings_pct"],
            details={"settlements": settlements, "obligation_ids": [r.id for r in rows]},
        )
        self.db.add(cycle)
        self.db.flush()
        now = datetime.combine(cycle_date, utcnow().time())
        for p, r, amount_base in result["settlements"]:
            payer, payee = members[p], members[r]
            debit = self._convert_logged(pool, amount_base, base, payer.currency)
            credit = self._convert_logged(pool, amount_base, base, payee.currency)
            self.rail.transfer(payer, payee, debit, credit, now)
        for row in rows:
            row.status = "settled"
            row.netting_cycle_id = cycle.id
        self.db.commit()
        summary["cycle_id"] = cycle.id
        return summary

    def _convert_logged(self, pool: CashPool, amount, frm: str, to: str) -> Decimal:
        out = self.fx.convert(amount, frm, to)
        if frm != to:
            self.db.add(
                FXConversion(
                    pool_id=pool.id,
                    from_currency=frm,
                    to_currency=to,
                    amount_in=amount,
                    amount_out=out,
                    rate=self.fx.rate(frm, to),
                    provider=self.fx.name,
                )
            )
        return out


def member_by_account(pool: CashPool, account_id: str) -> PoolMember:
    for m in pool.members:
        if m.account_id == account_id and m.is_active:
            return m
    raise ValueError(f"conta não pertence ao pool: {account_id}")
