from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import JSON, Boolean, Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models import utcnow

from .enums import MemberRole, NettingMode, PoolType, SweepDirection, SweepReason

JSONType = JSON().with_variant(JSONB(), "postgresql")
Money = Numeric(20, 4)


def _enum(cls):
    return SAEnum(cls, values_callable=lambda e: [m.value for m in e], native_enum=False, length=20)


class CashPool(Base):
    __tablename__ = "cash_pools"

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    pool_type: Mapped[PoolType] = mapped_column(_enum(PoolType), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    sweep_policy: Mapped[dict] = mapped_column(JSONType, default=dict)
    netting_policy: Mapped[dict] = mapped_column(JSONType, default=dict)
    yield_policy: Mapped[dict] = mapped_column(JSONType, default=dict)
    compliance_profile: Mapped[str] = mapped_column(String(2), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    members: Mapped[list["PoolMember"]] = relationship(back_populates="pool", cascade="all, delete-orphan", order_by="PoolMember.id")


class PoolMember(Base):
    __tablename__ = "pool_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    role: Mapped[MemberRole] = mapped_column(_enum(MemberRole), default=MemberRole.PARTICIPANT)
    sweep_direction: Mapped[SweepDirection] = mapped_column(_enum(SweepDirection), default=SweepDirection.BOTH)
    target_balance_rule: Mapped[dict] = mapped_column(JSONType, default=lambda: {"type": "zero"})
    min_balance: Mapped[Decimal] = mapped_column(Money, default=Decimal("0"))
    max_balance: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    pool: Mapped[CashPool] = relationship(back_populates="members")


class CashPosition(Base):
    __tablename__ = "cash_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("pool_members.id", ondelete="CASCADE"), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    balance: Mapped[Decimal] = mapped_column(Money, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="import")


class SweepInstruction(Base):
    __tablename__ = "sweep_instructions"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    from_account_id: Mapped[str] = mapped_column(String(64), nullable=False)
    to_account_id: Mapped[str] = mapped_column(String(64), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Money, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    reason: Mapped[SweepReason] = mapped_column(_enum(SweepReason))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    tax_flags: Mapped[list] = mapped_column(JSONType, default=list)
    as_of: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class IntercompanyObligation(Base):
    __tablename__ = "intercompany_obligations"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    payer_member_id: Mapped[int] = mapped_column(ForeignKey("pool_members.id", ondelete="CASCADE"))
    payee_member_id: Mapped[int] = mapped_column(ForeignKey("pool_members.id", ondelete="CASCADE"))
    amount: Mapped[Decimal] = mapped_column(Money, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    reference: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="open")
    netting_cycle_id: Mapped[int | None] = mapped_column(ForeignKey("netting_cycles.id"), nullable=True)


class NettingCycle(Base):
    __tablename__ = "netting_cycles"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    cycle_date: Mapped[date] = mapped_column(Date, nullable=False)
    mode: Mapped[NettingMode] = mapped_column(_enum(NettingMode))
    gross_volume: Mapped[Decimal] = mapped_column(Money, default=Decimal("0"))
    net_volume: Mapped[Decimal] = mapped_column(Money, default=Decimal("0"))
    savings_pct: Mapped[Decimal] = mapped_column(Numeric(6, 2), default=Decimal("0"))
    details: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class FXConversion(Base):
    __tablename__ = "fx_conversions"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    from_currency: Mapped[str] = mapped_column(String(3))
    to_currency: Mapped[str] = mapped_column(String(3))
    amount_in: Mapped[Decimal] = mapped_column(Money)
    amount_out: Mapped[Decimal] = mapped_column(Money)
    rate: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    provider: Mapped[str] = mapped_column(String(32))
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class YieldAccrual(Base):
    __tablename__ = "yield_accruals"

    id: Mapped[int] = mapped_column(primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("cash_pools.id", ondelete="CASCADE"), index=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("pool_members.id", ondelete="CASCADE"))
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    base_amount: Mapped[Decimal] = mapped_column(Money)
    rate: Mapped[Decimal] = mapped_column(Numeric(12, 8))
    accrued: Mapped[Decimal] = mapped_column(Money)
    currency: Mapped[str] = mapped_column(String(3))
