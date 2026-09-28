"""store empty JSON as SQL NULL

Before this, a failed run's raw (Python None) was stored as the JSON value 'null', so
`raw IS NULL` missed it and reextract tried to parse it. Data-only migration.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMNS = [("runs", "raw"), ("batches", "config_snapshot"), ("strategies", "evidence"),
           ("strategies", "output")]


def upgrade() -> None:
    pg = op.get_bind().dialect.name == "postgresql"
    for table, column in COLUMNS:
        if table == "batches":
            continue  # NOT NULL column; always holds an object
        cond = f"{column} = 'null'::jsonb" if pg else f"{column} = 'null'"
        op.execute(f"UPDATE {table} SET {column} = NULL WHERE {cond}")


def downgrade() -> None:
    pass  # nothing to undo: SQL NULL is what these values always meant
