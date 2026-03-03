import sqlite3
from pathlib import Path

from sourcetax import categorization, storage
from sourcetax.gold import filter_human_labeled_gold, is_human_labeled_gold_record


def test_is_human_labeled_gold_record():
    human = {
        "category_final": "Travel",
        "raw_payload": {"label_source": "human"},
    }
    non_human = {
        "category_final": "Travel",
        "raw_payload": {"label_source": "rules"},
    }
    unlabeled = {
        "category_final": "",
        "raw_payload": {"label_source": "human"},
    }
    assert is_human_labeled_gold_record(human) is True
    assert is_human_labeled_gold_record(non_human) is False
    assert is_human_labeled_gold_record(unlabeled) is False


def test_filter_human_labeled_gold():
    rows = [
        {"id": "1", "category_final": "Travel", "raw_payload": {"label_source": "human"}},
        {"id": "2", "category_final": "Travel", "raw_payload": {"label_source": "rules"}},
        {"id": "3", "category_final": "", "raw_payload": {"label_source": "human"}},
    ]
    filtered, skipped = filter_human_labeled_gold(rows)
    assert [r["id"] for r in filtered] == ["1"]
    assert skipped == 2


def test_save_category_override_marks_human_label_source(tmp_path: Path):
    db_path = tmp_path / "store.db"
    storage.ensure_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO canonical_records (
            id, source, source_record_id, transaction_date, merchant_raw, merchant_norm,
            amount, currency, direction, payment_method, category_pred, category_final,
            confidence, matched_transaction_id, match_score, evidence_keys, raw_payload, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "rec_human_1",
            "bank",
            "src_1",
            "2026-03-02",
            "TEST MERCHANT",
            "test merchant",
            10.0,
            "USD",
            "expense",
            "card",
            None,
            None,
            0.8,
            None,
            None,
            "[]",
            "{}",
            "[]",
        ),
    )
    conn.commit()
    conn.close()

    categorization.save_category_override("rec_human_1", "Travel", str(db_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT category_final, raw_payload FROM canonical_records WHERE id = ?", ("rec_human_1",))
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "Travel"
    assert '"label_source": "human"' in (row[1] or "")

