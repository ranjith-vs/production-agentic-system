"""Applies every *.sql file in SCHEMA_DIR, in filename order, against the
app's Postgres connection pool at startup. All statements are idempotent
(CREATE TABLE/INDEX IF NOT EXISTS, ADD COLUMN IF NOT EXISTS) so re-running
this on every boot is safe. Table structure lives entirely in schema/*.sql,
not in this file or anywhere else in the codebase - see schema/ at the repo
root to read or change it."""

import os
from pathlib import Path

SCHEMA_DIR = Path(os.environ.get("SCHEMA_DIR", "schema"))


async def apply_schema(pool) -> None:
    sql_files = sorted(SCHEMA_DIR.glob("*.sql"))
    async with pool.connection() as conn:
        for path in sql_files:
            await conn.execute(path.read_text())
