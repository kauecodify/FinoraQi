from decimal import ROUND_HALF_EVEN, Decimal

CENT = Decimal("0.01")
UNIT4 = Decimal("0.0001")


def d(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def q2(value) -> Decimal:
    return d(value).quantize(CENT, rounding=ROUND_HALF_EVEN)


def q4(value) -> Decimal:
    return d(value).quantize(UNIT4, rounding=ROUND_HALF_EVEN)
