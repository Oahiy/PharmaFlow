import os
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from psycopg import Connection, OperationalError

COMPOSE_DATABASE_URL = "postgresql://pharmaflow:pharmaflow@127.0.0.1:5432/pharmaflow"


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.getenv("DATABASE_URL", COMPOSE_DATABASE_URL)
    try:
        with Connection.connect(url, connect_timeout=5):
            pass
    except OperationalError as exc:
        pytest.fail(f"无法连接集成测试所需的 PostgreSQL：{exc}", pytrace=False)
    return url


@pytest.fixture
def run_config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid4()), "user_id": str(uuid4())}}
