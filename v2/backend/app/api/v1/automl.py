import uuid
from pathlib import Path

import joblib
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.uploads import read_upload
from app.core.config import get_settings
from app.core.deps import CurrentUser, DbSession
from app.core.feature_flags import require_flag
from app.core.jobs import jobs
from app.db.models import MLModelRecord
from app.db.session import SessionLocal
from app.ml import automl_engine, evolution
from app.ml.incremental import IncrementalTrainer
from app.processing.spreadsheet import clean, profile

router = APIRouter(prefix="/automl", tags=["automl"], dependencies=[require_flag("automl.enabled")])


class PredictIn(BaseModel):
    model_id: str
    rows: list[dict] = Field(min_length=1, max_length=10_000)


def _train_and_store(df, target: str, features: list[str] | None, problem_type: str | None, tenant: str) -> dict:
    result = automl_engine.train(df, target, features, problem_type)
    path = automl_engine.save(result, str(Path(get_settings().models_dir) / tenant))
    with SessionLocal() as db:
        db.add(
            MLModelRecord(
                id=result.model_id,
                tenant_id=tenant,
                kind=f"automl:{result.best_model}",
                problem_type=result.problem_type,
                target=target,
                features=result.features,
                metrics={"leaderboard": result.leaderboard, "holdout": result.holdout},
                path=path,
            )
        )
        db.commit()
    return {
        "model_id": result.model_id,
        "problem_type": result.problem_type,
        "best_model": result.best_model,
        "features": result.features,
        "leaderboard": result.leaderboard,
        "holdout": result.holdout,
    }


def _features(raw: str | None) -> list[str] | None:
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


@router.post("/analyze")
async def analyze(user: CurrentUser, file: UploadFile = File(...)):
    df, report = clean(await read_upload(file))
    prof = profile(df)
    suggestions = [{"column": c, "problem_type": automl_engine.infer_problem_type(df[c])} for c in df.columns if df[c].nunique() > 1]
    return {"report": report, "profile": prof, "target_suggestions": suggestions}


@router.post("/train")
async def train(
    user: CurrentUser,
    file: UploadFile = File(...),
    target: str = Form(...),
    features: str | None = Form(None),
    problem_type: str | None = Form(None),
    background: bool = Form(False),
):
    if problem_type not in (None, "", "classification", "regression"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "problem_type inválido")
    df, _ = clean(await read_upload(file))
    args = (df, target, _features(features), problem_type or None, user.tenant_id)
    if background:
        return {"job_id": jobs.submit(_train_and_store, *args), "status": "PENDING"}
    try:
        return _train_and_store(*args)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/evolve")
async def evolve(
    user: CurrentUser, file: UploadFile = File(...), target: str = Form(...), generations: int = Form(3), population: int = Form(6)
):
    if not (1 <= generations <= 20 and 4 <= population <= 30):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "generations 1-20 e population 4-30")
    df, _ = clean(await read_upload(file))
    if target not in df.columns:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"coluna alvo '{target}' não encontrada")
    return {"job_id": jobs.submit(evolution.evolve, df, target, generations, population), "status": "PENDING"}


@router.post("/incremental")
async def incremental(
    user: CurrentUser,
    file: UploadFile = File(...),
    target: str = Form(...),
    problem_type: str = Form(...),
    batch_size: int = Form(256),
    epochs: int = Form(3),
):
    df, _ = clean(await read_upload(file))
    if target not in df.columns:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"coluna alvo '{target}' não encontrada")
    features = [c for c in df.columns if c != target]
    categorical = [c for c in features if df[c].dtype == object]
    try:
        trainer = IncrementalTrainer(problem_type, features, categorical)
        metrics = trainer.fit_stream(df, target, batch_size=max(8, batch_size), epochs=max(1, min(epochs, 20)))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    model_id = str(uuid.uuid4())
    directory = Path(get_settings().models_dir) / user.tenant_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_id}.incremental.joblib"
    joblib.dump(trainer, path)
    return {"model_id": model_id, "n_seen": trainer.n_seen, "metrics": metrics, "history": trainer.history[-20:]}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, user: CurrentUser):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job não encontrado")
    return job


@router.get("/models")
def list_models(user: CurrentUser, db: DbSession):
    rows = db.scalars(
        select(MLModelRecord).where(MLModelRecord.tenant_id == user.tenant_id).order_by(MLModelRecord.created_at.desc())
    ).all()
    return [
        {
            "model_id": r.id,
            "kind": r.kind,
            "problem_type": r.problem_type,
            "target": r.target,
            "features": r.features,
            "metrics": r.metrics,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


def _record(db, tenant: str, model_id: str) -> MLModelRecord:
    rec = db.get(MLModelRecord, model_id)
    if rec is None or rec.tenant_id != tenant or not Path(rec.path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "modelo não encontrado")
    return rec


@router.post("/predict")
def predict(payload: PredictIn, user: CurrentUser, db: DbSession):
    rec = _record(db, user.tenant_id, payload.model_id)
    bundle = automl_engine.load(rec.path)
    try:
        return {"model_id": rec.id, "predictions": automl_engine.predict(bundle, payload.rows)}
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"falha na predição: {exc}") from exc


@router.get("/models/{model_id}/download")
def download(model_id: str, user: CurrentUser, db: DbSession):
    rec = _record(db, user.tenant_id, model_id)
    return FileResponse(rec.path, filename=f"{model_id}.joblib", media_type="application/octet-stream")
