"""Camada de câmbio com provedor estático (padrão, determinístico) e HTTP com fallback."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from decimal import Decimal

from .money import d, q4

DEFAULT_USD_RATES = {
    "USD": "1",
    "BRL": "5.40",
    "EUR": "0.92",
    "MXN": "18.10",
    "INR": "83.50",
    "GBP": "0.79",
}


class FXError(ValueError):
    pass


class FXProvider(ABC):
    name = "abstract"

    @abstractmethod
    def rate(self, frm: str, to: str) -> Decimal: ...

    def convert(self, amount: Decimal, frm: str, to: str) -> Decimal:
        if frm == to:
            return q4(amount)
        return q4(d(amount) * self.rate(frm, to))


class StaticFXProvider(FXProvider):
    """Taxas de referência por USD. Usado em desenvolvimento, testes e como fallback."""

    name = "static"

    def __init__(self, usd_rates: dict[str, str] | None = None) -> None:
        self.usd_rates = {k: d(v) for k, v in (usd_rates or DEFAULT_USD_RATES).items()}

    def rate(self, frm: str, to: str) -> Decimal:
        if frm == to:
            return Decimal("1")
        try:
            return (self.usd_rates[to] / self.usd_rates[frm]).quantize(Decimal("0.00000001"))
        except KeyError as exc:
            raise FXError(f"par de moedas não suportado: {frm}/{to}") from exc


class HttpFXProvider(FXProvider):
    """Frankfurter (BCE). Cache com TTL e fallback para StaticFXProvider em caso de falha."""

    name = "frankfurter"

    def __init__(self, ttl_seconds: int = 3600, fallback: FXProvider | None = None, timeout: float = 5.0) -> None:
        self.ttl = ttl_seconds
        self.fallback = fallback or StaticFXProvider()
        self.timeout = timeout
        self._cache: dict[tuple[str, str], tuple[float, Decimal]] = {}
        self._lock = threading.Lock()

    def rate(self, frm: str, to: str) -> Decimal:
        if frm == to:
            return Decimal("1")
        key = (frm, to)
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > time.monotonic():
                return hit[1]
        try:
            import httpx

            resp = httpx.get("https://api.frankfurter.app/latest", params={"from": frm, "to": to}, timeout=self.timeout)
            resp.raise_for_status()
            value = d(resp.json()["rates"][to])
        except Exception:
            return self.fallback.rate(frm, to)
        with self._lock:
            self._cache[key] = (time.monotonic() + self.ttl, value)
        return value


def get_fx_provider(kind: str = "static") -> FXProvider:
    return HttpFXProvider() if kind == "http" else StaticFXProvider()
