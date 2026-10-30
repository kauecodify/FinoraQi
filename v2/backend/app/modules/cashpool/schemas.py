from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import ComplianceProfile, MemberRole, NettingMode, PoolType, SweepDirection

Currency = Field(pattern=r"^[A-Z]{3}$")


class MemberIn(BaseModel):
    entity_id: str = Field(min_length=1, max_length=64)
    account_id: str = Field(min_length=1, max_length=64)
    currency: str = Currency
    role: MemberRole = MemberRole.PARTICIPANT
    sweep_direction: SweepDirection = SweepDirection.BOTH
    target_balance_rule: dict = Field(default_factory=lambda: {"type": "zero"})
    min_balance: Decimal = Field(default=Decimal("0"), ge=0)
    max_balance: Decimal | None = Field(default=None, ge=0)

    @field_validator("target_balance_rule")
    @classmethod
    def _rule(cls, v: dict) -> dict:
        kind = v.get("type", "zero")
        if kind not in ("zero", "fixed", "forecast"):
            raise ValueError("target_balance_rule.type deve ser zero, fixed ou forecast")
        if kind == "fixed" and "amount" not in v:
            raise ValueError("regra fixed exige amount")
        return {**v, "type": kind}


class PoolCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    pool_type: PoolType = PoolType.PHYSICAL
    base_currency: str = Currency
    compliance_profile: ComplianceProfile = ComplianceProfile.BR
    sweep_policy: dict = Field(default_factory=lambda: {"min_sweep_amount": "1.00"})
    netting_policy: dict = Field(default_factory=lambda: {"mode": "multilateral"})
    yield_policy: dict = Field(default_factory=lambda: {"method": "pro_rata"})
    members: list[MemberIn] = Field(min_length=1)

    @field_validator("netting_policy")
    @classmethod
    def _netting(cls, v: dict) -> dict:
        NettingMode(v.get("mode", "none"))
        return v

    @field_validator("yield_policy")
    @classmethod
    def _yield(cls, v: dict) -> dict:
        if v.get("method", "pro_rata") not in ("pro_rata", "tiered", "contract"):
            raise ValueError("yield_policy.method inválido")
        return v


class MemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    entity_id: str
    account_id: str
    currency: str
    role: MemberRole
    sweep_direction: SweepDirection
    target_balance_rule: dict
    min_balance: Decimal
    max_balance: Decimal | None
    is_active: bool


class PoolOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    tenant_id: str
    name: str
    pool_type: PoolType
    base_currency: str
    compliance_profile: str
    sweep_policy: dict
    netting_policy: dict
    yield_policy: dict
    is_active: bool
    created_at: datetime
    members: list[MemberOut]


class PositionIn(BaseModel):
    account_id: str
    balance: Decimal
    as_of: datetime


class PositionsImport(BaseModel):
    positions: list[PositionIn] = Field(min_length=1, max_length=10_000)


class PositionOut(BaseModel):
    member_id: int
    entity_id: str
    account_id: str
    role: MemberRole
    currency: str
    balance: Decimal
    balance_base: Decimal
    base_currency: str


class PositionsOut(BaseModel):
    as_of: datetime
    base_currency: str
    total_base: Decimal
    positions: list[PositionOut]


class ObligationIn(BaseModel):
    payer_account_id: str
    payee_account_id: str
    amount: Decimal = Field(gt=0)
    currency: str = Currency
    due_date: date
    reference: str | None = Field(default=None, max_length=64)


class ObligationsImport(BaseModel):
    obligations: list[ObligationIn] = Field(min_length=1, max_length=10_000)


class NettingRun(BaseModel):
    cycle_date: date
    mode: NettingMode | None = None
    dry_run: bool = False


class YieldRun(BaseModel):
    period_start: date
    period_end: date
    pool_yield: Decimal = Field(gt=0)
    dry_run: bool = False
