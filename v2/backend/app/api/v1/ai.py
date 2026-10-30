from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.ai.categorizer import get_categorizer
from app.ai.insights import generate_insights
from app.core.deps import CurrentUser, DbSession
from app.db.models import Transaction
from app.ml.forecast import forecast_series

router = APIRouter(prefix="/ai", tags=["ai"])


class CategorizeIn(BaseModel):
    descriptions: list[str] = Field(min_length=1, max_length=1000)


class FeedbackIn(BaseModel):
    description: str = Field(min_length=1, max_length=255)
    category: str


class ForecastIn(BaseModel):
    dates: list[date] = Field(min_length=2, max_length=5000)
    values: list[float] = Field(min_length=2, max_length=5000)
    horizon: int = Field(default=7, ge=1, le=365)


@router.get("/categories")
def categories(user: CurrentUser):
    return get_categorizer().categories


@router.post("/categorize")
def categorize(payload: CategorizeIn, user: CurrentUser):
    return get_categorizer().predict(payload.descriptions)


@router.post("/categorize/feedback")
def feedback(payload: FeedbackIn, user: CurrentUser):
    try:
        get_categorizer().learn(payload.description, payload.category)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"ok": True}


@router.get("/insights")
def insights(user: CurrentUser, db: DbSession, horizon: int = Query(30, ge=1, le=180)):
    rows = [
        {"date": t.date, "type": t.type, "amount": float(t.amount), "category": t.category, "description": t.description}
        for t in db.scalars(select(Transaction).where(Transaction.user_id == user.id))
    ]
    return generate_insights(rows, horizon)


@router.post("/forecast")
def forecast(payload: ForecastIn, user: CurrentUser):
    if len(payload.dates) != len(payload.values):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "dates e values devem ter o mesmo tamanho")
    return forecast_series(payload.dates, payload.values, payload.horizon)
