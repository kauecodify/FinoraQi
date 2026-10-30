"""Processamento de planilhas: leitura segura, perfil, limpeza, agregação e exportação."""

from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd

ALLOWED = {".csv", ".xlsx", ".xls"}


class DataError(ValueError):
    pass


def _snake(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text or "col"


def read_table(filename: str, content: bytes, max_rows: int = 200_000) -> pd.DataFrame:
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED:
        raise DataError(f"formato não suportado: {ext or 'sem extensão'} (use CSV ou XLSX)")
    if not content:
        raise DataError("arquivo vazio")
    try:
        if ext == ".csv":
            text = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    text = content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", nrows=max_rows)
        else:
            df = pd.read_excel(io.BytesIO(content), nrows=max_rows)
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"não foi possível ler o arquivo: {exc}") from exc
    if df.empty or df.shape[1] == 0:
        raise DataError("planilha sem dados")
    return df


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    s = series.astype(str).str.strip().str.replace(r"^R\$\s*", "", regex=True)
    br = s.str.match(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$") | s.str.match(r"^-?\d+,\d+$")
    s = s.where(~br, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    converted = pd.to_numeric(s, errors="coerce")
    non_null = series.notna() & (series.astype(str).str.strip() != "")
    if non_null.sum() and converted[non_null].notna().mean() >= 0.9:
        return converted
    return series


def _coerce_datetime(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    sample = series.dropna().astype(str).head(50)
    if sample.empty or not sample.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() >= 0.9:
        return series
    dayfirst = sample.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").mean() >= 0.5
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return parsed if parsed.notna().mean() >= 0.9 else series


def clean(df: pd.DataFrame, drop_duplicates: bool = True, fill_strategy: str = "median") -> tuple[pd.DataFrame, dict]:
    if fill_strategy not in ("median", "mean", "zero", "none"):
        raise DataError("fill_strategy inválido")
    report: dict = {"rows_in": int(len(df))}
    out = df.copy()
    out.columns = pd.Index(_dedupe([_snake(c) for c in out.columns]))
    out = out.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string").str.strip().replace("", pd.NA).astype(object)
        out[col] = _coerce_datetime(_coerce_numeric(out[col]))
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        report["duplicates_removed"] = int(before - len(out))
    filled = {}
    if fill_strategy != "none":
        for col in out.select_dtypes(include=[np.number]).columns:
            n = int(out[col].isna().sum())
            if n:
                value = {"median": out[col].median(), "mean": out[col].mean(), "zero": 0}[fill_strategy]
                out[col] = out[col].fillna(0 if pd.isna(value) else value)
                filled[col] = n
    report.update({"rows_out": int(len(out)), "filled": filled, "columns": list(out.columns)})
    return out.reset_index(drop=True), report


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def profile(df: pd.DataFrame) -> dict:
    cols = []
    for col in df.columns:
        s = df[col]
        info = {"name": str(col), "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            info.update(
                {
                    "min": _f(desc.get("min")),
                    "max": _f(desc.get("max")),
                    "mean": _f(desc.get("mean")),
                    "std": _f(desc.get("std")),
                    "outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr == iqr else 0,
                }
            )
        else:
            info["top"] = {str(k): int(v) for k, v in s.astype(str).value_counts().head(5).items()}
        cols.append(info)
    return {"rows": int(len(df)), "columns": cols}


def _f(v) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 4)


def aggregate(df: pd.DataFrame, group_by: list[str], metrics: dict[str, str]) -> pd.DataFrame:
    allowed = {"sum", "mean", "min", "max", "count", "median"}
    for col in group_by + list(metrics):
        if col not in df.columns:
            raise DataError(f"coluna inexistente: {col}")
    for fn in metrics.values():
        if fn not in allowed:
            raise DataError(f"agregação não permitida: {fn}")
    return df.groupby(group_by, dropna=False).agg(metrics).reset_index()


def to_records(df: pd.DataFrame, limit: int = 100) -> list[dict]:
    head = df.head(limit).copy()
    for col in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[col]):
            head[col] = head[col].dt.strftime("%Y-%m-%d")
    return head.astype(object).where(head.notna(), None).to_dict(orient="records")


def export(df: pd.DataFrame, fmt: str) -> tuple[bytes, str]:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv"
    if fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    raise DataError("formato de exportação deve ser csv ou xlsx")
