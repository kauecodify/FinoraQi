from enum import Enum


class PoolType(str, Enum):
    PHYSICAL = "physical"
    NOTIONAL = "notional"
    HYBRID = "hybrid"


class MemberRole(str, Enum):
    HEADER = "header"
    PARTICIPANT = "participant"


class SweepDirection(str, Enum):
    IN = "in"
    OUT = "out"
    BOTH = "both"


class SweepReason(str, Enum):
    ZERO = "zero_balance"
    TARGET = "target_balance"
    FORECAST = "forecast"
    MANUAL = "manual"


class NettingMode(str, Enum):
    NONE = "none"
    BILATERAL = "bilateral"
    MULTILATERAL = "multilateral"


class YieldMethod(str, Enum):
    PRO_RATA = "pro_rata"
    TIERED = "tiered"
    CONTRACT = "contract"


class ComplianceProfile(str, Enum):
    BR = "BR"
    US = "US"
    EU = "EU"
    MX = "MX"
    IN = "IN"
