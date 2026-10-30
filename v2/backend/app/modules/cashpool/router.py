from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import CurrentTenant, DbSession
from app.core.feature_flags import require_flag
from app.db.models import utcnow

from .fx import FXError
from .schemas import NettingRun, ObligationsImport, PoolCreate, PoolOut, PositionsImport, PositionsOut, YieldRun
from .service import CashPoolService, NotFound

router = APIRouter(prefix="/cashpool", tags=["cashpool"], dependencies=[require_flag("cashpool.enabled")])


def _call(fn, *args):
    try:
        return fn(*args)
    except NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (ValueError, FXError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/pools", response_model=PoolOut, status_code=201)
def create_pool(payload: PoolCreate, db: DbSession, tenant: CurrentTenant):
    return _call(CashPoolService(db, tenant).create_pool, payload)


@router.get("/pools", response_model=list[PoolOut])
def list_pools(db: DbSession, tenant: CurrentTenant):
    return CashPoolService(db, tenant).list_pools()


@router.get("/pools/{pool_id}", response_model=PoolOut)
def get_pool(pool_id: int, db: DbSession, tenant: CurrentTenant):
    return _call(CashPoolService(db, tenant).get_pool, pool_id)


@router.post("/pools/{pool_id}/positions", status_code=201)
def import_positions(pool_id: int, payload: PositionsImport, db: DbSession, tenant: CurrentTenant):
    return {"imported": _call(CashPoolService(db, tenant).import_positions, pool_id, payload)}


@router.get("/pools/{pool_id}/positions", response_model=PositionsOut)
def positions(pool_id: int, db: DbSession, tenant: CurrentTenant, as_of: datetime | None = None):
    return _call(CashPoolService(db, tenant).positions, pool_id, as_of or utcnow())


@router.post("/pools/{pool_id}/sweep", dependencies=[require_flag("cashpool.sweep")])
def run_sweep(pool_id: int, db: DbSession, tenant: CurrentTenant, dry_run: bool = False, as_of: datetime | None = None):
    return _call(CashPoolService(db, tenant).run_sweep, pool_id, as_of or utcnow(), dry_run)


@router.post("/pools/{pool_id}/obligations", status_code=201, dependencies=[require_flag("cashpool.netting")])
def add_obligations(pool_id: int, payload: ObligationsImport, db: DbSession, tenant: CurrentTenant):
    return {"imported": _call(CashPoolService(db, tenant).add_obligations, pool_id, payload)}


@router.post("/pools/{pool_id}/netting", dependencies=[require_flag("cashpool.netting")])
def run_netting(pool_id: int, payload: NettingRun, db: DbSession, tenant: CurrentTenant):
    return _call(CashPoolService(db, tenant).run_netting, pool_id, payload.cycle_date, payload.mode, payload.dry_run)


@router.post("/pools/{pool_id}/yield", dependencies=[require_flag("cashpool.yield")])
def allocate_yield(pool_id: int, payload: YieldRun, db: DbSession, tenant: CurrentTenant):
    return _call(
        CashPoolService(db, tenant).allocate_yield, pool_id, payload.period_start, payload.period_end, payload.pool_yield, payload.dry_run
    )


@router.get("/pools/{pool_id}/forecast/{member_id}")
def forecast(pool_id: int, member_id: int, db: DbSession, tenant: CurrentTenant, horizon: int = Query(7, ge=1, le=90)):
    return _call(CashPoolService(db, tenant).forecast, pool_id, member_id, horizon)


@router.get("/pools/{pool_id}/history")
def history(pool_id: int, db: DbSession, tenant: CurrentTenant, limit: int = Query(50, ge=1, le=500)):
    return _call(CashPoolService(db, tenant).history, pool_id, limit)
