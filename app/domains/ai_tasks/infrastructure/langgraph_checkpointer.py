"""Checkpoint store for the LangGraph research agent."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

# PyCharm's package-requirements inspection does not always see Poetry deps.
# noinspection PyPackageRequirements
from langgraph.checkpoint.base import BaseCheckpointSaver

# noinspection PyPackageRequirements
from langgraph.checkpoint.memory import InMemorySaver

# noinspection PyPackageRequirements
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def open_checkpointer(postgres_url: str) -> AsyncIterator[BaseCheckpointSaver]:
    """Postgres checkpointer for the URL, or an in-memory one if the URL is empty."""
    if not postgres_url:
        yield InMemorySaver()
        return

    # AsyncPostgresSaver requires autocommit, no prepared statements, and dict rows.
    async with AsyncConnectionPool[AsyncConnection[DictRow]](
        postgres_url,
        connection_class=AsyncConnection[DictRow],
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer
