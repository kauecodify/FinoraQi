from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.db.models import FeatureFlag, User
from app.db.session import get_db


def is_enabled(db: Session, tenant_id: str, key: str) -> bool:
    row = db.scalar(select(FeatureFlag).where(FeatureFlag.tenant_id == tenant_id, FeatureFlag.key == key))
    if row is not None:
        return row.enabled
    return get_settings().default_feature_flags.get(key, False)


def all_flags(db: Session, tenant_id: str) -> dict[str, bool]:
    flags = dict(get_settings().default_feature_flags)
    for row in db.scalars(select(FeatureFlag).where(FeatureFlag.tenant_id == tenant_id)):
        flags[row.key] = row.enabled
    return flags


def set_flag(db: Session, tenant_id: str, key: str, enabled: bool) -> None:
    row = db.scalar(select(FeatureFlag).where(FeatureFlag.tenant_id == tenant_id, FeatureFlag.key == key))
    if row is None:
        db.add(FeatureFlag(tenant_id=tenant_id, key=key, enabled=enabled))
    else:
        row.enabled = enabled
    db.commit()


def require_flag(*keys: str):
    def checker(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
        for key in keys:
            if not is_enabled(db, user.tenant_id, key):
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"Recurso desabilitado: {key}")

    return Depends(checker)
