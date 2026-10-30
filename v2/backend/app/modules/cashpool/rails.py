"""Payment rails. LedgerRail é o rail interno: lê e grava saldos em cash_positions,
de modo que sweep e netting tenham efeito auditável mesmo sem banco externo conectado."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CashPosition, PoolMember


class PaymentRail(ABC):
    @abstractmethod
    def get_balance(self, member: PoolMember, as_of: datetime | None = None) -> Decimal: ...

    @abstractmethod
    def transfer(self, debit: PoolMember, credit: PoolMember, debit_amount: Decimal, credit_amount: Decimal, as_of: datetime) -> None: ...


class LedgerRail(PaymentRail):
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_balance(self, member: PoolMember, as_of: datetime | None = None) -> Decimal:
        stmt = select(CashPosition).where(CashPosition.member_id == member.id)
        if as_of is not None:
            stmt = stmt.where(CashPosition.as_of <= as_of)
        row = self.db.scalars(stmt.order_by(CashPosition.as_of.desc(), CashPosition.id.desc()).limit(1)).first()
        return Decimal(row.balance) if row else Decimal("0")

    def _post(self, member: PoolMember, delta: Decimal, as_of: datetime) -> None:
        balance = self.get_balance(member) + delta
        self.db.add(
            CashPosition(
                pool_id=member.pool_id, member_id=member.id, as_of=as_of, balance=balance, currency=member.currency, source="ledger"
            )
        )
        self.db.flush()

    def transfer(self, debit: PoolMember, credit: PoolMember, debit_amount: Decimal, credit_amount: Decimal, as_of: datetime) -> None:
        if debit_amount <= 0 or credit_amount <= 0:
            raise ValueError("valor de transferência deve ser positivo")
        self._post(debit, -debit_amount, as_of)
        self._post(credit, credit_amount, as_of)
