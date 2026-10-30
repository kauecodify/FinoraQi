from datetime import date
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.ai.categorizer import get_categorizer
from app.core.deps import CurrentUser, DbSession
from app.db.models import Goal, Investment, Transaction, Wallet

router = APIRouter(tags=["finance"])


class TransactionIn(BaseModel):
    type: Literal["income", "expense"]
    amount: Decimal = Field(gt=0, max_digits=16, decimal_places=2)
    category: str | None = Field(default=None, max_length=64)
    description: str | None = Field(default=None, max_length=255)
    date: date


class TransactionOut(TransactionIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    category_source: str


class InvestmentIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    amount: Decimal = Field(gt=0, max_digits=16, decimal_places=2)
    rate: float = Field(default=0.0, ge=0, le=1000)
    type: str | None = Field(default=None, max_length=32)
    date: date


class InvestmentOut(InvestmentIn):
    model_config = ConfigDict(from_attributes=True)
    id: int


class GoalIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    target: Decimal = Field(gt=0, max_digits=16, decimal_places=2)
    saved: Decimal = Field(default=Decimal("0"), ge=0, max_digits=16, decimal_places=2)
    deadline: date | None = None


class GoalOut(GoalIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    progress: float = 0.0


class WalletIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: str | None = Field(default=None, max_length=32)
    balance: Decimal = Field(default=Decimal("0"), max_digits=16, decimal_places=2)
    rate: float = Field(default=0.0, ge=0, le=1000)
    institution: str | None = Field(default=None, max_length=120)


class WalletOut(WalletIn):
    model_config = ConfigDict(from_attributes=True)
    id: int


def _owned(db, model, item_id: int, user_id: int):
    obj = db.get(model, item_id)
    if obj is None or obj.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Registro não encontrado")
    return obj


def _goal_out(g: Goal) -> GoalOut:
    out = GoalOut.model_validate(g)
    out.progress = round(float(g.saved / g.target), 4) if g.target else 0.0
    return out


def _apply_category(data: dict) -> dict:
    if not data.get("category"):
        text = data.get("description") or ""
        if text.strip():
            data["category"] = get_categorizer().predict([text])[0]["category"]
            data["category_source"] = "ai"
        else:
            data["category"] = "outros"
            data["category_source"] = "default"
    else:
        data["category_source"] = "user"
    return data


@router.get("/transactions", response_model=list[TransactionOut])
def list_transactions(
    user: CurrentUser,
    db: DbSession,
    type: Literal["income", "expense"] | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = Query(500, ge=1, le=5000),
):
    stmt = select(Transaction).where(Transaction.user_id == user.id)
    if type:
        stmt = stmt.where(Transaction.type == type)
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    return db.scalars(stmt.order_by(Transaction.date.desc(), Transaction.id.desc()).limit(limit)).all()


@router.post("/transactions", response_model=TransactionOut, status_code=201)
def create_transaction(payload: TransactionIn, user: CurrentUser, db: DbSession):
    tx = Transaction(user_id=user.id, **_apply_category(payload.model_dump()))
    db.add(tx)
    db.commit()
    return tx


@router.put("/transactions/{item_id}", response_model=TransactionOut)
def update_transaction(item_id: int, payload: TransactionIn, user: CurrentUser, db: DbSession):
    tx = _owned(db, Transaction, item_id, user.id)
    for k, v in _apply_category(payload.model_dump()).items():
        setattr(tx, k, v)
    db.commit()
    return tx


@router.delete("/transactions/{item_id}", status_code=204)
def delete_transaction(item_id: int, user: CurrentUser, db: DbSession):
    db.delete(_owned(db, Transaction, item_id, user.id))
    db.commit()


@router.get("/investments", response_model=list[InvestmentOut])
def list_investments(user: CurrentUser, db: DbSession):
    return db.scalars(select(Investment).where(Investment.user_id == user.id).order_by(Investment.date.desc())).all()


@router.post("/investments", response_model=InvestmentOut, status_code=201)
def create_investment(payload: InvestmentIn, user: CurrentUser, db: DbSession):
    obj = Investment(user_id=user.id, **payload.model_dump())
    db.add(obj)
    db.commit()
    return obj


@router.put("/investments/{item_id}", response_model=InvestmentOut)
def update_investment(item_id: int, payload: InvestmentIn, user: CurrentUser, db: DbSession):
    obj = _owned(db, Investment, item_id, user.id)
    for k, v in payload.model_dump().items():
        setattr(obj, k, v)
    db.commit()
    return obj


@router.delete("/investments/{item_id}", status_code=204)
def delete_investment(item_id: int, user: CurrentUser, db: DbSession):
    db.delete(_owned(db, Investment, item_id, user.id))
    db.commit()


@router.get("/goals", response_model=list[GoalOut])
def list_goals(user: CurrentUser, db: DbSession):
    return [_goal_out(g) for g in db.scalars(select(Goal).where(Goal.user_id == user.id).order_by(Goal.id))]


@router.post("/goals", response_model=GoalOut, status_code=201)
def create_goal(payload: GoalIn, user: CurrentUser, db: DbSession):
    obj = Goal(user_id=user.id, **payload.model_dump())
    db.add(obj)
    db.commit()
    return _goal_out(obj)


@router.put("/goals/{item_id}", response_model=GoalOut)
def update_goal(item_id: int, payload: GoalIn, user: CurrentUser, db: DbSession):
    obj = _owned(db, Goal, item_id, user.id)
    for k, v in payload.model_dump().items():
        setattr(obj, k, v)
    db.commit()
    return _goal_out(obj)


@router.delete("/goals/{item_id}", status_code=204)
def delete_goal(item_id: int, user: CurrentUser, db: DbSession):
    db.delete(_owned(db, Goal, item_id, user.id))
    db.commit()


@router.get("/wallets", response_model=list[WalletOut])
def list_wallets(user: CurrentUser, db: DbSession):
    return db.scalars(select(Wallet).where(Wallet.user_id == user.id).order_by(Wallet.id)).all()


@router.post("/wallets", response_model=WalletOut, status_code=201)
def create_wallet(payload: WalletIn, user: CurrentUser, db: DbSession):
    obj = Wallet(user_id=user.id, **payload.model_dump())
    db.add(obj)
    db.commit()
    return obj


@router.put("/wallets/{item_id}", response_model=WalletOut)
def update_wallet(item_id: int, payload: WalletIn, user: CurrentUser, db: DbSession):
    obj = _owned(db, Wallet, item_id, user.id)
    for k, v in payload.model_dump().items():
        setattr(obj, k, v)
    db.commit()
    return obj


@router.delete("/wallets/{item_id}", status_code=204)
def delete_wallet(item_id: int, user: CurrentUser, db: DbSession):
    db.delete(_owned(db, Wallet, item_id, user.id))
    db.commit()


@router.get("/dashboard/summary")
def dashboard_summary(user: CurrentUser, db: DbSession):
    txs = db.scalars(select(Transaction).where(Transaction.user_id == user.id)).all()
    income = sum((t.amount for t in txs if t.type == "income"), Decimal("0"))
    expense = sum((t.amount for t in txs if t.type == "expense"), Decimal("0"))
    invested = sum((i.amount for i in db.scalars(select(Investment).where(Investment.user_id == user.id))), Decimal("0"))
    wallets = sum((w.balance for w in db.scalars(select(Wallet).where(Wallet.user_id == user.id))), Decimal("0"))
    return {
        "income": str(income),
        "expense": str(expense),
        "balance": str(income - expense),
        "invested": str(invested),
        "wallets": str(wallets),
        "transactions": len(txs),
    }
