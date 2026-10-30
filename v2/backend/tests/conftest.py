import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="finoraqi-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["MODELS_DIR"] = f"{_tmp}/models"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-0123456789abcdef0123456789"
os.environ["FRONTEND_DIR"] = f"{_tmp}/no-frontend"

import itertools  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

_counter = itertools.count()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def register(client, name="Teste"):
    n = next(_counter)
    r = client.post("/api/v1/auth/register", json={"nome": name, "email": f"user{n}@finoraqi.com.br", "senha": "senha-segura-123"})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def auth(client):
    data = register(client)
    return {"Authorization": f"Bearer {data['token']}"}
