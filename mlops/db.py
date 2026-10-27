"""PostgreSQL: connection and schema (psycopg 3). Rows come back as dicts."""

from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from mlops.config import database_url

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(url=None):
    return psycopg.connect(url or database_url(), row_factory=dict_row, autocommit=True)


@contextmanager
def connection(url=None):
    conn = connect(url)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(url=None):
    """Create the tables that do not exist yet (every start of the ui and of the worker)."""
    with connection(url) as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))


def one(conn, query, *args):
    return conn.execute(query, args).fetchone()


def all_rows(conn, query, *args):
    return conn.execute(query, args).fetchall()


def json(value):
    """A Python value for a jsonb column."""
    return Jsonb(value)
