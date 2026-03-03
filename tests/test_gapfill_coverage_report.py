import sqlite3
from pathlib import Path


def test_gapfill_coverage_query_shape(tmp_path: Path):
    from tools.gapfill_coverage_report import _fetch_counts

    db_path = tmp_path / "staging.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE staging_transactions (
            rowid INTEGER PRIMARY KEY,
            source TEXT,
            category_external TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO staging_transactions (source, category_external) VALUES (?, ?)",
        [
            ("synthetic_gapfill", "COGS"),
            ("synthetic_gapfill", "COGS"),
            ("synthetic_gapfill", "Insurance"),
            ("hf_mitulshah", "Food & Dining"),
        ],
    )
    conn.commit()

    counts = _fetch_counts(conn)
    conn.close()

    assert counts[0][0] == "COGS"
    assert counts[0][1] == 2
    assert ("Insurance", 1) in counts
    assert all(cat != "Food & Dining" for cat, _ in counts)

