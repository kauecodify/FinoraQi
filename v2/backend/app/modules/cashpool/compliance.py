"""Regras por jurisdição. Bloqueios estruturais retornam erros; eventos fiscais viram flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


class MemberLike(Protocol):
    role: str
    currency: str


class BaseRules:
    min_members = 2

    def validate_pool_creation(self, members: list, base_currency: str) -> ValidationResult:
        errors = []
        if len(members) < self.min_members:
            errors.append(f"pool precisa de ao menos {self.min_members} membros")
        headers = [m for m in members if _role(m) == "header"]
        if len(headers) != 1:
            errors.append("pool precisa de exatamente uma conta header")
        accounts = [m.account_id for m in members]
        if len(accounts) != len(set(accounts)):
            errors.append("account_id duplicado no pool")
        return ValidationResult(not errors, errors)

    def sweep_flags(self, amount_base: Decimal, base_currency: str) -> list[str]:
        return []

    def validate_sweep(self, amount_base: Decimal, base_currency: str) -> ValidationResult:
        if amount_base <= 0:
            return ValidationResult(False, ["valor de sweep deve ser positivo"])
        return ValidationResult(True, flags=self.sweep_flags(amount_base, base_currency))


class BrazilRules(BaseRules):
    """IOF sobre mútuo intercompany, preços de transferência."""

    IOF_THRESHOLD_BRL = Decimal("30000")

    def sweep_flags(self, amount_base, base_currency):
        flags = ["transfer_pricing_doc"]
        if base_currency == "BRL" and amount_base > self.IOF_THRESHOLD_BRL:
            flags.append("iof_0.38pct")
        return flags


class MexicoRules(BrazilRules):
    def sweep_flags(self, amount_base, base_currency):
        return ["transfer_pricing_doc"]


class EURules(BaseRules):
    """ATAD: limite de dedutibilidade de juros (30% EBITDA) e retenção na fonte."""

    def sweep_flags(self, amount_base, base_currency):
        return ["atad_interest_limit", "withholding_tax_review"]


class USRules(BaseRules):
    """Section 385: documentação debt/equity para mútuos relevantes."""

    SECTION_385_THRESHOLD = Decimal("1000000")

    def sweep_flags(self, amount_base, base_currency):
        return ["section_385_doc"] if amount_base >= self.SECTION_385_THRESHOLD else []


class IndiaRules(EURules):
    def sweep_flags(self, amount_base, base_currency):
        return ["rbi_ecb_review", "withholding_tax_review"]


RULES: dict[str, type[BaseRules]] = {"BR": BrazilRules, "MX": MexicoRules, "EU": EURules, "US": USRules, "IN": IndiaRules}


def _role(member) -> str:
    role = getattr(member, "role", "")
    return getattr(role, "value", role)


class ComplianceEngine:
    def __init__(self, profile: str) -> None:
        self.rules = RULES.get(profile, BaseRules)()

    def validate_pool_creation(self, members: list, base_currency: str) -> ValidationResult:
        return self.rules.validate_pool_creation(members, base_currency)

    def validate_sweep(self, amount_base: Decimal, base_currency: str) -> ValidationResult:
        return self.rules.validate_sweep(amount_base, base_currency)
