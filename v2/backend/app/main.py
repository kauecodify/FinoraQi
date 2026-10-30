from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1 import ai, auth, automl, data, finance, flags, payments, score
from app.core.config import get_settings
from app.core.jobs import jobs
from app.core.websocket import manager
from app.db.session import init_db
from app.ml.scoring import NUMBA_AVAILABLE, warmup
from app.modules.cashpool.router import router as cashpool_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.auto_create_tables:
        init_db()
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
    warmup()
    yield
    await manager.close_all()
    jobs.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api/v1")
    for module in (auth, finance, payments, score, flags, data, ai, automl):
        api.include_router(module.router)
    api.include_router(cashpool_router)

    @api.get("/health", tags=["health"])
    def health():
        return {"status": "healthy", "version": settings.app_version, "numba": NUMBA_AVAILABLE}

    app.include_router(api)

    @app.get("/health", include_in_schema=False)
    def root_health():
        return {"status": "healthy", "version": settings.app_version}

    frontend = Path(settings.frontend_dir)
    if not frontend.is_absolute():
        frontend = Path(__file__).resolve().parent.parent / frontend
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
