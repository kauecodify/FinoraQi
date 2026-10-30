import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from app.core.deps import CurrentUser, DbSession
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User

router = APIRouter(tags=["auth"])


class RegisterIn(BaseModel):
    nome: str = Field(min_length=2, max_length=120)
    email: EmailStr
    senha: str = Field(min_length=8, max_length=128)
    telefone: str | None = Field(default=None, max_length=32)


class LoginIn(BaseModel):
    email: EmailStr
    senha: str


class UserOut(BaseModel):
    id: int
    nome: str
    email: str
    telefone: str | None
    tenant_id: str
    role: str


class TokenOut(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserOut


class ProfileUpdate(BaseModel):
    nome: str | None = Field(default=None, min_length=2, max_length=120)
    telefone: str | None = Field(default=None, max_length=32)


def _out(user: User) -> UserOut:
    return UserOut(id=user.id, nome=user.nome, email=user.email, telefone=user.telefone, tenant_id=user.tenant_id, role=user.role)


def _token(user: User) -> TokenOut:
    return TokenOut(token=create_access_token(str(user.id), user.tenant_id, user.role), user=_out(user))


@router.post("/auth/register", response_model=TokenOut, status_code=201)
def register(payload: RegisterIn, db: DbSession):
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email já cadastrado")
    user = User(
        tenant_id=f"t_{uuid.uuid4().hex[:12]}",
        nome=payload.nome,
        email=email,
        telefone=payload.telefone,
        password_hash=hash_password(payload.senha),
        role="admin",
    )
    db.add(user)
    db.commit()
    return _token(user)


@router.post("/auth/login", response_model=TokenOut)
def login(payload: LoginIn, db: DbSession):
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.senha, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciais inválidas")
    return _token(user)


@router.get("/profile", response_model=UserOut)
def get_profile(user: CurrentUser):
    return _out(user)


@router.put("/profile", response_model=UserOut)
def update_profile(payload: ProfileUpdate, user: CurrentUser, db: DbSession):
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(user, key, value)
    db.commit()
    return _out(user)
