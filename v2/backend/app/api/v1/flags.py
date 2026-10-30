from fastapi import APIRouter
from pydantic import BaseModel

from app.core.deps import AdminUser, CurrentUser, DbSession
from app.core.feature_flags import all_flags, set_flag

router = APIRouter(tags=["flags"])


class FlagIn(BaseModel):
    enabled: bool


@router.get("/flags")
def list_flags(user: CurrentUser, db: DbSession):
    return all_flags(db, user.tenant_id)


@router.put("/flags/{key}")
def update_flag(key: str, payload: FlagIn, user: AdminUser, db: DbSession):
    set_flag(db, user.tenant_id, key, payload.enabled)
    return all_flags(db, user.tenant_id)
