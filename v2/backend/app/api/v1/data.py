import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.api.uploads import read_upload
from app.core.deps import CurrentUser
from app.processing.spreadsheet import DataError, aggregate, clean, export, profile, to_records

router = APIRouter(prefix="/data", tags=["data"])


def _bad(exc: Exception):
    raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/profile")
async def profile_file(user: CurrentUser, file: UploadFile = File(...)):
    df = await read_upload(file)
    return {"filename": file.filename, **profile(df), "preview": to_records(df, 20)}


@router.post("/clean")
async def clean_file(
    user: CurrentUser, file: UploadFile = File(...), drop_duplicates: bool = Form(True), fill_strategy: str = Form("median")
):
    df = await read_upload(file)
    try:
        out, report = clean(df, drop_duplicates, fill_strategy)
    except DataError as exc:
        _bad(exc)
    return {"report": report, "profile": profile(out), "preview": to_records(out, 50)}


@router.post("/aggregate")
async def aggregate_file(user: CurrentUser, file: UploadFile = File(...), group_by: str = Form(...), metrics: str = Form(...)):
    df, _ = clean(await read_upload(file))
    try:
        metric_map = json.loads(metrics)
        if not isinstance(metric_map, dict):
            raise DataError("metrics deve ser um objeto JSON {coluna: agregação}")
        out = aggregate(df, [c.strip() for c in group_by.split(",") if c.strip()], metric_map)
    except (DataError, json.JSONDecodeError) as exc:
        _bad(exc)
    return {"rows": len(out), "data": to_records(out, 1000)}


@router.post("/export")
async def export_file(user: CurrentUser, file: UploadFile = File(...), fmt: str = Form("csv")):
    df, _ = clean(await read_upload(file))
    try:
        content, media = export(df, fmt)
    except DataError as exc:
        _bad(exc)
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="finoraqi_clean.{fmt}"'})
