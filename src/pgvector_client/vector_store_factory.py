import os
import time
import random
import logging
from threading import Lock

from psycopg import OperationalError as PsycopgOperationalError
from sqlalchemy.exc import OperationalError as SAOperationalError
from langchain_postgres import PGVector

logger = logging.getLogger(__name__)

_vector_store = None
_vector_store_lock = Lock()


def _build_connection_string():
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "49605")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "Passw0rd")
    postgres_db = os.getenv("POSTGRES_DB", "demodb01")

    return (
        f"postgresql+psycopg://{postgres_user}:{postgres_password}"
        f"@{postgres_host}:{postgres_port}/{postgres_db}?client_encoding=utf8"
    )


def _create_vector_store(
    embeddings,
    collection_name="posts",
    max_retries=5,
    initial_delay=2,
    max_delay=30,
):
    connection = _build_connection_string()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            store = PGVector(
                embeddings=embeddings,
                collection_name=collection_name,
                connection=connection,
                engine_args={
                    "pool_pre_ping": True,
                    "pool_recycle": 300,
                },
                # create_extension=False,
            )
            store.create_collection()
            return store
        except (PsycopgOperationalError, SAOperationalError) as e:
            last_error = e
            if attempt == max_retries:
                break

            delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
            sleep_time = delay + random.uniform(0, delay * 0.5)
            logger.warning(
                "PGVector init failed (attempt %s/%s), retrying in %.2fs: %s",
                attempt,
                max_retries,
                sleep_time,
                e,
            )
            time.sleep(sleep_time)

    raise RuntimeError(
        f"Could not initialize PGVector after {max_retries} attempts"
    ) from last_error


def get_vector_store(
    embeddings,
    collection_name="posts",
    force_reconnect=False,
):
    global _vector_store

    if force_reconnect:
        with _vector_store_lock:
            _vector_store = None

    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                _vector_store = _create_vector_store(
                    embeddings=embeddings,
                    collection_name=collection_name,
                )

    return _vector_store