"""
SourceTax Review UI (Streamlit).

Workflow:
1. Upload / ingest
2. Review grid
3. Exceptions / review queue
4. Export
"""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sourcetax import categorization, exporter, ingest, matching, reconciliation, storage, taxonomy
from sourcetax.review_logic import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    build_issue_flags,
    confidence_value,
    has_missing_required_fields,
    merchant_ambiguity_reason,
    primary_issue_type,
)


DB_PATH = "data/store.db"
GOLD_PATH = "data/gold/gold_transactions.jsonl"
UPLOAD_DIR = Path("tmp/ui_uploads")


st.set_page_config(page_title="SourceTax Review", page_icon=None, layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .main .block-container {max-width: 1350px; padding-top: 1rem; padding-bottom: 2rem;}
          .app-title {font-size: 1.9rem; font-weight: 800; margin-bottom: 0.2rem;}
          .app-sub {color: #5b6773; margin-bottom: 0.9rem;}
          div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,0.12); border-radius: 12px; padding: 0.75rem;}
          .stButton>button {border-radius: 10px; padding: 0.55rem 0.8rem;}
          div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden; border: 1px solid rgba(49,51,63,0.12);}
          div[data-testid="stExpander"] {border-radius: 12px; border: 1px solid rgba(49,51,63,0.10);}
          .badge {display:inline-block; padding: 0.12rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700;}
          .badge-high {background:#d9f5ea; color:#0b6b46;}
          .badge-mid {background:#fff1cc; color:#7a5a00;}
          .badge-low {background:#ffe0de; color:#9d1c1c;}
          .badge-issue {background:#e8f0fe; color:#174ea6;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default
    return default


def fmt_money(value: Any) -> str:
    try:
        if value is None or value == "":
            return "-"
        return f"${float(value):,.2f}"
    except Exception:
        return str(value)


def fmt_pct(value: Any) -> str:
    try:
        if value is None or value == "":
            return "-"
        return f"{float(value):.1%}"
    except Exception:
        return str(value)


def confidence_level(value: Any) -> str:
    v = confidence_value(value, default=-1.0)
    if v < 0.0:
        return "unknown"
    if v >= 0.85:
        return "high"
    if v >= 0.60:
        return "medium"
    return "low"


def confidence_badge_html(value: Any) -> str:
    level = confidence_level(value)
    if level == "unknown":
        return "<span class='badge badge-issue'>Unknown</span>"
    cls = {"high": "badge-high", "medium": "badge-mid", "low": "badge-low"}[level]
    return f"<span class='badge {cls}'>{level.title()} ({fmt_pct(value)})</span>"


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def get_conn() -> sqlite3.Connection:
    storage.ensure_db(Path(DB_PATH))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def reset_workspace() -> None:
    storage.reset_db(Path(DB_PATH))
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    st.session_state.pop("ingest_results", None)


def query_rows(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def record_lookup_parts(record_ref: str) -> tuple[str, Any]:
    text = str(record_ref or "").strip()
    if text.startswith("rowid:"):
        try:
            return "rowid", int(text.split(":", 1)[1])
        except Exception:
            return "id", text
    return "id", text


def parse_record_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    rec["raw_payload"] = safe_json_loads(rec.get("raw_payload"), {})
    rec["evidence_keys"] = safe_json_loads(rec.get("evidence_keys"), [])
    rec["tags"] = safe_json_loads(rec.get("tags"), [])
    return rec


def fetch_record(record_id: str) -> Optional[Dict[str, Any]]:
    key, value = record_lookup_parts(record_id)
    if key == "rowid":
        rows = query_rows("SELECT rowid, * FROM canonical_records WHERE rowid = ?", (value,))
    else:
        rows = query_rows("SELECT rowid, * FROM canonical_records WHERE id = ?", (value,))
    return parse_record_fields(rows[0]) if rows else None


def fetch_all_records_df() -> pd.DataFrame:
    rows = query_rows(
        """
        SELECT rowid, id, source, source_record_id, transaction_date, merchant_raw, merchant_norm,
               amount, direction, category_pred, category_final, confidence,
               matched_transaction_id, match_score, raw_payload, evidence_keys
        FROM canonical_records
        ORDER BY transaction_date DESC, rowid DESC
        """
    )
    parsed = [parse_record_fields(r) for r in rows]
    return pd.DataFrame(parsed) if parsed else pd.DataFrame()


def get_run_metadata() -> Dict[str, str]:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit": git_commit_hash(),
        "db": DB_PATH,
    }


def dashboard_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_records": 0,
            "receipts": 0,
            "matched_receipts": 0,
            "match_rate": 0.0,
            "categorization_coverage": 0.0,
            "avg_confidence": None,
            "needs_review": 0,
        }
    receipts_mask = df["source"].eq("receipt")
    receipts = int(receipts_mask.sum())
    matched_receipts = int((receipts_mask & df["matched_transaction_id"].notna()).sum())
    covered = int(((df["category_final"].fillna("") != "") | (df["category_pred"].fillna("") != "")).sum())
    avg_conf = pd.to_numeric(df["confidence"], errors="coerce").mean()
    recon = reconciliation.summary_metrics(DB_PATH)
    needs_review = (
        recon.get("low_confidence_queue_size", 0)
        + recon.get("conflicts_queue_size", 0)
        + max(receipts - matched_receipts, 0)
    )
    return {
        "total_records": len(df),
        "receipts": receipts,
        "matched_receipts": matched_receipts,
        "match_rate": (matched_receipts / receipts) if receipts else 0.0,
        "categorization_coverage": covered / len(df) if len(df) else 0.0,
        "avg_confidence": float(avg_conf) if pd.notna(avg_conf) else None,
        "needs_review": int(needs_review),
    }


def label_source_for_record(rec: Dict[str, Any]) -> str:
    raw = rec.get("raw_payload") if isinstance(rec.get("raw_payload"), dict) else {}
    if rec.get("category_final"):
        return "human"
    if isinstance(raw, dict):
        if raw.get("ensemble_decision"):
            return "ensemble"
        if raw.get("ml_prediction") or raw.get("model_pred"):
            return "ml"
    return "rules"


def effective_category_for_row(rec: Dict[str, Any]) -> str:
    return (
        taxonomy.normalize_category_name(rec.get("category_final"))
        or taxonomy.normalize_category_name(rec.get("category_pred"))
        or "Uncategorized"
    )


def reason_source_for_row(rec: Dict[str, Any]) -> str:
    raw = rec.get("raw_payload") if isinstance(rec.get("raw_payload"), dict) else {}
    reasons = raw.get("rule_reason") if isinstance(raw, dict) else None
    if isinstance(reasons, list) and reasons:
        return str(reasons[0])
    if isinstance(raw, dict) and raw.get("ml_prediction"):
        return "ml_prediction"
    return label_source_for_record(rec)


def build_review_queue_df(all_df: pd.DataFrame) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()
    low_conf = {r["id"]: r for r in reconciliation.low_confidence_categorizations(DB_PATH)}
    conflicts = {r["id"]: r for r in reconciliation.conflicts_queue(DB_PATH)}
    unmatched_receipts = {r["id"]: r for r in reconciliation.unmatched_receipts(DB_PATH)}
    unmatched_bank = {r["id"]: r for r in reconciliation.unmatched_bank_transactions(DB_PATH)}
    has_receipts = bool((all_df["source"].fillna("").astype(str).str.lower() == "receipt").any())
    rows = []
    for _, row in all_df.iterrows():
        rid = row.get("id")
        if not rid:
            continue
        parsed = parse_record_fields(row.to_dict())
        issue_flags = build_issue_flags(
            parsed,
            has_receipts=has_receipts,
            is_conflict=rid in conflicts,
            low_conf_threshold=LOW_CONFIDENCE_THRESHOLD,
        )
        if rid in unmatched_receipts:
            issue_flags.setdefault("unmatched_receipt", "receipt not matched to a transaction")
        if has_receipts and rid in unmatched_bank:
            issue_flags.setdefault("unmatched_bank_txn", "bank/POS transaction not linked to a receipt")
        if not issue_flags:
            continue
        raw_payload = parsed.get("raw_payload") if isinstance(parsed.get("raw_payload"), dict) else {}
        issue = primary_issue_type(issue_flags) or "low_confidence"
        rows.append(
            {
                "id": rid,
                "transaction_date": parsed.get("transaction_date"),
                "merchant_raw": parsed.get("merchant_raw"),
                "merchant_norm": parsed.get("merchant_norm"),
                "amount": parsed.get("amount"),
                "source": parsed.get("source"),
                "predicted_category": parsed.get("category_pred"),
                "final_category": parsed.get("category_final"),
                "effective_category": effective_category_for_row(parsed),
                "confidence": confidence_value(parsed.get("confidence")),
                "issue_type": issue,
                "issue_types": ", ".join(issue_flags.keys()),
                "issue_notes": " | ".join(issue_flags.values()),
                "rules_pred": conflicts.get(rid, {}).get("rules_pred"),
                "ml_pred": conflicts.get(rid, {}).get("ml_pred")
                or (raw_payload.get("ml_prediction") if isinstance(raw_payload, dict) else None),
                "label_source": label_source_for_record(
                    {"category_final": parsed.get("category_final"), "raw_payload": raw_payload}
                ),
            }
        )
    q = pd.DataFrame(rows)
    if not q.empty:
        q = q.sort_values(by=["transaction_date", "confidence"], ascending=[False, True], na_position="last")
    return q


def build_review_grid_df(all_df: pd.DataFrame) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()
    has_receipts = bool((all_df["source"].fillna("").astype(str).str.lower() == "receipt").any())
    conflict_ids = {r["id"] for r in reconciliation.conflicts_queue(DB_PATH)}
    rows = []
    for _, row in all_df.iterrows():
        rec = parse_record_fields(row.to_dict())
        ui_id = rec.get("id") or f"rowid:{rec.get('rowid')}"
        confidence_num = confidence_value(rec.get("confidence"))
        raw_payload = rec.get("raw_payload") if isinstance(rec.get("raw_payload"), dict) else {}
        rule_conf = confidence_value(raw_payload.get("rule_confidence"))
        ml_conf_raw = raw_payload.get("ml_confidence")
        ml_conf = confidence_value(ml_conf_raw) if ml_conf_raw is not None else None
        saved_category = taxonomy.normalize_category_name(rec.get("category_final")) or ""
        predicted_category = taxonomy.normalize_category_name(rec.get("category_pred")) or "Uncategorized"
        selected_category = saved_category or predicted_category
        if saved_category:
            confidence_source = "human"
        elif raw_payload.get("rule_category") and taxonomy.normalize_category_name(raw_payload.get("rule_category")) == predicted_category:
            confidence_source = "rule"
        elif raw_payload.get("ml_prediction") and taxonomy.normalize_category_name(raw_payload.get("ml_prediction")) == predicted_category:
            confidence_source = "ml"
        else:
            confidence_source = "pipeline"
        issue_flags = build_issue_flags(
            rec,
            has_receipts=has_receipts,
            is_conflict=str(rec.get("id") or "") in conflict_ids,
            low_conf_threshold=LOW_CONFIDENCE_THRESHOLD,
        )
        rows.append(
            {
                "id": ui_id,
                "date": rec.get("transaction_date"),
                "merchant_raw": rec.get("merchant_raw"),
                "merchant_normalized": rec.get("merchant_norm"),
                "amount": rec.get("amount"),
                "source": rec.get("source"),
                "predicted_category": predicted_category,
                "saved_category": saved_category,
                "selected_category": selected_category,
                "effective_category": effective_category_for_row(rec),
                "confidence": confidence_num,
                "confidence_band": confidence_level(confidence_num).title(),
                "confidence_source": confidence_source,
                "rule_confidence": rule_conf,
                "ml_confidence": ml_conf,
                "reason_source": reason_source_for_row(rec),
                "approval_status": "Approved" if saved_category else "Pending",
                "approved": bool(rec.get("category_final")),
                "unknown_merchant": bool(merchant_ambiguity_reason(rec)),
                "merchant_issue": merchant_ambiguity_reason(rec),
                "missing_fields": has_missing_required_fields(rec),
                "issue_summary": " | ".join(issue_flags.values()),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["date", "merchant_raw"], ascending=[False, True], na_position="last")
    return df


def category_options() -> List[str]:
    options = taxonomy.load_sourcetax_categories(include_uncategorized=False)
    return ["Uncategorized"] + options if options else ["Uncategorized"]


def detect_uploaded_source(filename: str, raw_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".pdf"}:
        return "receipt"
    if suffix != ".csv":
        raise ValueError(f"Unsupported file type: {suffix}")
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    headers = {str(x or "").strip().lower() for x in (reader.fieldnames or [])}
    if {"order_id", "location", "total"}.issubset(headers):
        return "toast"
    if {"date", "description", "amount"}.issubset(headers):
        return "bank"
    if {"date", "amount"}.issubset(headers) and ("payee" in headers or "description" in headers):
        return "quickbooks"
    if {"merchant", "date", "total"}.issubset(headers):
        return "receipt"
    raise ValueError("Could not detect source from headers.")


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = Path(uploaded_file.name).name
    target = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    target.write_bytes(uploaded_file.getvalue())
    return target


def process_uploaded_files(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        raw_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()
        row = {
            "file_name": uploaded_file.name,
            "source_detected": "",
            "rows_found": 0,
            "parsed_ok": 0,
            "parsed_failed": 0,
            "status": "ok",
            "notes": "",
        }
        try:
            source = detect_uploaded_source(uploaded_file.name, raw_bytes)
            row["source_detected"] = source
            saved_path = save_uploaded_file(uploaded_file)
            if source == "receipt" and suffix in {".png", ".jpg", ".jpeg", ".pdf"}:
                ok = ingest.ingest_receipt_file(saved_path, db_path=DB_PATH)
                row["rows_found"] = 1
                row["parsed_ok"] = 1 if ok else 0
                row["parsed_failed"] = 0 if ok else 1
                row["notes"] = "Receipt OCR parsed" if ok else "Receipt parse failed"
            else:
                text = raw_bytes.decode("utf-8-sig")
                row_count = max(sum(1 for _ in csv.DictReader(text.splitlines())), 0)
                inserted = ingest.ingest_and_store(str(saved_path), source, db_path=DB_PATH)
                row["rows_found"] = row_count
                row["parsed_ok"] = inserted
                row["parsed_failed"] = max(row_count - inserted, 0)
                row["notes"] = f"Ingested {inserted} rows"
        except Exception as exc:
            row["status"] = "error"
            row["parsed_failed"] = row["rows_found"] or 1
            row["notes"] = str(exc)
        results.append(row)
    return results


def run_processing_pipeline() -> Dict[str, int]:
    matched = matching.match_all_receipts(DB_PATH)
    categorized = categorization.categorize_all_records(DB_PATH)
    return {"matched_receipts": int(matched), "categorized_records": int(categorized)}


def save_category_override(record_id: str, category: str, notes: str = "Saved from review grid") -> None:
    to_save = "Other Expense" if category == "Uncategorized" else category
    key, value = record_lookup_parts(record_id)
    if key == "id":
        categorization.save_category_override(
            str(value),
            to_save,
            DB_PATH,
            label_confidence="medium",
            label_notes=notes,
        )
        return

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT raw_payload FROM canonical_records WHERE rowid = ?", (value,))
    row = cur.fetchone()
    raw_payload = safe_json_loads(row["raw_payload"] if row else {}, {})
    if not isinstance(raw_payload, dict):
        raw_payload = {}
    raw_payload["label_source"] = "human"
    raw_payload["label_confidence"] = "medium"
    raw_payload["label_notes"] = notes
    raw_payload["labeled_at_utc"] = datetime.utcnow().isoformat() + "Z"
    cur.execute(
        "UPDATE canonical_records SET category_final = ?, raw_payload = ? WHERE rowid = ?",
        (to_save, json.dumps(raw_payload, ensure_ascii=False), value),
    )
    conn.commit()
    conn.close()


def render_header() -> None:
    meta = get_run_metadata()
    st.markdown("<div class='app-title'>SourceTax</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-sub'>Upload data, review categories, resolve exceptions, export accounting outputs.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Run metadata"):
        st.write(f"DB: `{meta['db']}`")
        st.write(f"Commit: `{meta['commit']}`")
        st.write(f"Timestamp: `{meta['timestamp']}`")


def render_record_detail_panel(record: Dict[str, Any], queue_row: Optional[Dict[str, Any]] = None) -> None:
    rec = parse_record_fields(record)
    c1, c2 = st.columns([1.0, 1.15])
    with c1:
        with st.container(border=True):
            st.subheader("Transaction")
            st.write(f"Date: `{rec.get('transaction_date') or '-'}`")
            st.write(f"Merchant: **{rec.get('merchant_raw') or '-'}**")
            st.write(f"Normalized: `{rec.get('merchant_norm') or '-'}`")
            st.write(f"Amount: {fmt_money(rec.get('amount'))}")
            st.write(f"Source: `{rec.get('source') or '-'}`")
            st.markdown(confidence_badge_html(rec.get("confidence")), unsafe_allow_html=True)
            st.write(f"Predicted: **{taxonomy.normalize_category_name(rec.get('category_pred')) or 'Uncategorized'}**")
            st.write(f"Final: **{effective_category_for_row(rec)}**")
            st.write(f"Reason / source: `{reason_source_for_row(rec)}`")
            if rec.get("matched_transaction_id"):
                st.write(
                    f"Matched: `{rec.get('matched_transaction_id')}` ({fmt_pct(rec.get('match_score'))})"
                )
    with c2:
        with st.container(border=True):
            st.subheader("Evidence")
            for ev in (rec.get("evidence_keys") or [])[:10]:
                st.write(f"- `{ev}`")
            raw_payload = rec.get("raw_payload") or {}
            if isinstance(raw_payload, dict):
                if raw_payload.get("ocr_text"):
                    st.text_area(
                        "Receipt excerpt",
                        str(raw_payload.get("ocr_text"))[:800],
                        height=180,
                        disabled=True,
                    )
                if queue_row and queue_row.get("issue_type") == "conflict":
                    st.write(f"Rules prediction: `{queue_row.get('rules_pred') or '-'}`")
                    st.write(f"ML prediction: `{queue_row.get('ml_pred') or '-'}`")


def render_ingest_screen(all_df: pd.DataFrame) -> None:
    st.header("Upload / Ingest")
    stats = dashboard_stats(all_df)
    top = st.columns(4)
    top[0].metric("Rows in workspace", stats["total_records"])
    top[1].metric("Receipts", stats["receipts"])
    top[2].metric("Matched receipts", stats["matched_receipts"])
    top[3].metric("Needs review", stats["needs_review"])

    with st.container(border=True):
        st.subheader("Add files")
        uploaded_files = st.file_uploader(
            "Drop bank CSVs, card exports, receipt files, or POS exports",
            type=["csv", "png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Ingest uploaded files", use_container_width=True, disabled=not uploaded_files):
                results = process_uploaded_files(list(uploaded_files or []))
                st.session_state["ingest_results"] = results
                st.success(f"Processed {len(results)} file(s).")
                st.rerun()
        with c2:
            if st.button("Run matching + categorization", use_container_width=True):
                result = run_processing_pipeline()
                st.success(
                    f"Matched {result['matched_receipts']} receipts and categorized {result['categorized_records']} records."
                )
                st.rerun()
        with c3:
            if st.button("Reset workspace", use_container_width=True):
                reset_workspace()
                st.success("Cleared the review workspace. You can now upload a fresh batch.")
                st.rerun()

    with st.container(border=True):
        st.subheader("Ingest summary")
        ingest_results = st.session_state.get("ingest_results", [])
        if ingest_results:
            summary_df = pd.DataFrame(ingest_results)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            sources = sorted({row["source_detected"] for row in ingest_results if row.get("source_detected")})
            st.write(f"Sources detected: `{', '.join(sources) if sources else '-'}`")
        else:
            st.info("No uploads processed yet.")


def render_review_grid(all_df: pd.DataFrame) -> None:
    st.header("Review Grid")
    grid_df = build_review_grid_df(all_df)
    if grid_df.empty:
        st.info("No records loaded yet. Upload files on the ingest screen first.")
        return

    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns([1.0, 1.0, 1.0, 1.0, 1.8])
        low_conf_only = f1.checkbox("Low confidence only")
        unknown_only = f2.checkbox("Ambiguous merchant")
        unapproved_only = f3.checkbox("Unapproved only")
        source_options = sorted(grid_df["source"].dropna().unique().tolist())
        selected_sources = f4.multiselect("Source", source_options, default=source_options)
        search_term = f5.text_input("Search merchant", placeholder="Starbucks, Uber, Amazon...")

    filtered = grid_df.copy()
    if low_conf_only:
        filtered = filtered[filtered["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    if unknown_only:
        filtered = filtered[filtered["unknown_merchant"]]
    if unapproved_only:
        filtered = filtered[~filtered["approved"]]
    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]
    if search_term.strip():
        term = search_term.strip().lower()
        filtered = filtered[
            filtered["merchant_raw"].fillna("").str.lower().str.contains(term)
            | filtered["merchant_normalized"].fillna("").str.lower().str.contains(term)
        ]

    if filtered.empty:
        st.info("No rows match the current review filters.")
        return

    action_cols = st.columns([1.2, 1.2, 2.4])
    with action_cols[0]:
        candidates = filtered[(filtered["confidence"] >= HIGH_CONFIDENCE_THRESHOLD) & (~filtered["approved"])]
        if st.button(
            f"Bulk approve high-confidence ({len(candidates)})",
            use_container_width=True,
            disabled=candidates.empty,
        ):
            saved = 0
            for _, row in candidates.iterrows():
                predicted_category = str(row["predicted_category"] or "").strip()
                if predicted_category in {"", "Uncategorized"}:
                    continue
                save_category_override(str(row["id"]), predicted_category, notes="Bulk-approved from review grid")
                saved += 1
            st.success(f"Approved {saved} high-confidence rows.")
            st.rerun()
    with action_cols[1]:
        if st.button("Export reviewed labels", use_container_width=True):
            result = exporter.export_gold_transactions_jsonl(DB_PATH, GOLD_PATH, append=True)
            st.success(f"Exported {result['exported']} labels. Gold total: {result['total_after']}")
    with action_cols[2]:
        st.caption("This is the main work surface: edit final category inline, tick approve, then save.")

    editor_df = filtered[
        [
            "id",
            "date",
            "merchant_raw",
            "merchant_normalized",
            "amount",
            "predicted_category",
            "saved_category",
            "effective_category",
            "confidence",
            "confidence_band",
            "confidence_source",
            "rule_confidence",
            "ml_confidence",
            "reason_source",
            "approval_status",
            "issue_summary",
            "selected_category",
        ]
    ].copy()
    editor_df["confidence_pct"] = (pd.to_numeric(editor_df["confidence"], errors="coerce").fillna(0.0) * 100).round().astype(int)
    editor_df["rule_confidence_pct"] = (pd.to_numeric(editor_df["rule_confidence"], errors="coerce").fillna(0.0) * 100).round().astype(int)
    editor_df["ml_confidence_pct"] = (pd.to_numeric(editor_df["ml_confidence"], errors="coerce").fillna(0.0) * 100).round().astype(int)
    editor_df["approve"] = False
    original_selected_categories = dict(zip(editor_df["id"].astype(str), editor_df["selected_category"].astype(str)))

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "id": st.column_config.TextColumn("ID"),
            "date": st.column_config.TextColumn("Date"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "confidence_pct": st.column_config.ProgressColumn(
                "Pipeline Conf",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "rule_confidence_pct": st.column_config.ProgressColumn(
                "Rule Conf",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "ml_confidence_pct": st.column_config.ProgressColumn(
                "ML Conf",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "confidence_band": st.column_config.TextColumn("Band"),
            "saved_category": st.column_config.TextColumn("Saved Category"),
            "effective_category": st.column_config.TextColumn("Effective Category"),
            "confidence_source": st.column_config.TextColumn("Conf Source"),
            "approval_status": st.column_config.TextColumn("Status"),
            "selected_category": st.column_config.SelectboxColumn(
                "Category To Save",
                options=category_options(),
                required=True,
            ),
            "approve": st.column_config.CheckboxColumn("Approve"),
        },
        key="review_grid_editor",
        column_order=[
            "id",
            "date",
            "merchant_raw",
            "merchant_normalized",
            "amount",
            "predicted_category",
            "saved_category",
            "effective_category",
            "confidence_pct",
            "rule_confidence_pct",
            "ml_confidence_pct",
            "confidence_band",
            "confidence_source",
            "reason_source",
            "approval_status",
            "issue_summary",
            "selected_category",
            "approve",
        ],
    )

    if st.button("Save grid changes", use_container_width=True):
        saved = 0
        for _, row in edited.iterrows():
            changed = str(row.get("selected_category")) != original_selected_categories.get(str(row.get("id")), "")
            if bool(row.get("approve")) or changed:
                save_category_override(str(row["id"]), str(row["selected_category"]))
                saved += 1
        st.success(f"Saved {saved} approval(s).")
        st.rerun()

    detail_choices = filtered["id"].astype(str).tolist()
    detail_labels = {}
    for _, row in filtered.iterrows():
        rid = str(row.get("id"))
        merchant = str(row.get("merchant_raw") or "Unknown merchant")
        detail_labels[rid] = f"{rid} • {merchant}"
    detail_id = st.selectbox(
        "Open row details",
        detail_choices,
        format_func=lambda rid: detail_labels.get(str(rid), str(rid)),
        key="review_detail_id",
    )
    if detail_id:
        record = fetch_record(str(detail_id))
        if record:
            render_record_detail_panel(record)


def render_exceptions_screen(all_df: pd.DataFrame) -> None:
    st.header("Exceptions / Review Queue")
    queue_df = build_review_queue_df(all_df)
    grid_df = build_review_grid_df(all_df)

    low_conf = (
        queue_df[queue_df["issue_types"].fillna("").str.contains("low_confidence")].copy()
        if not queue_df.empty
        else pd.DataFrame()
    )
    conflicts = (
        queue_df[queue_df["issue_types"].fillna("").str.contains("conflict")].copy()
        if not queue_df.empty
        else pd.DataFrame()
    )
    unmatched_receipts = pd.DataFrame(reconciliation.unmatched_receipts(DB_PATH))
    ambiguous_merchants = grid_df[grid_df["merchant_issue"].fillna("").astype(str) != ""].copy() if not grid_df.empty else pd.DataFrame()
    missing_fields = grid_df[grid_df["missing_fields"]].copy() if not grid_df.empty else pd.DataFrame()

    stats = st.columns(5)
    stats[0].metric("Low confidence", len(low_conf))
    stats[1].metric("Conflicts", len(conflicts))
    stats[2].metric("Unmatched receipts", len(unmatched_receipts))
    stats[3].metric("Ambiguous merchants", len(ambiguous_merchants))
    stats[4].metric("Missing fields", len(missing_fields))

    tabs = st.tabs(
        [
            "Low Confidence",
            "Conflicts",
            "Unmatched Receipts",
            "Ambiguous Merchants",
            "Missing Fields",
        ]
    )

    with tabs[0]:
        if low_conf.empty:
            st.success("No low-confidence rows.")
        else:
            st.dataframe(low_conf, use_container_width=True, hide_index=True)
    with tabs[1]:
        if conflicts.empty:
            st.success("No conflicts right now.")
        else:
            st.dataframe(conflicts, use_container_width=True, hide_index=True)
    with tabs[2]:
        if unmatched_receipts.empty:
            st.success("No unmatched receipts.")
        else:
            st.dataframe(unmatched_receipts, use_container_width=True, hide_index=True)
    with tabs[3]:
        if ambiguous_merchants.empty:
            st.success("No ambiguous merchants.")
        else:
            st.dataframe(
                ambiguous_merchants[["id", "date", "merchant_raw", "merchant_normalized", "amount", "source", "merchant_issue"]],
                use_container_width=True,
                hide_index=True,
            )
    with tabs[4]:
        if missing_fields.empty:
            st.success("No rows with missing required fields.")
        else:
            st.dataframe(
                missing_fields[["id", "date", "merchant_raw", "amount", "source"]],
                use_container_width=True,
                hide_index=True,
            )


@dataclass
class ExportCard:
    label: str
    path: Path


def count_file_rows(path: Path) -> Optional[int]:
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                n = -1
                for _ in reader:
                    n += 1
                return max(n, 0)
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
    except Exception:
        return None
    return None


def render_export_card(card: ExportCard) -> None:
    with st.container(border=True):
        st.subheader(card.label)
        st.write(f"File: `{card.path}`")
        if not card.path.exists():
            st.caption("Not generated yet")
            return
        row_count = count_file_rows(card.path)
        if row_count is not None:
            st.write(f"Rows: {row_count}")
        st.write(
            f"Updated: {datetime.fromtimestamp(card.path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            data = card.path.read_bytes()
            st.download_button("Download", data=data, file_name=card.path.name, use_container_width=True)
        except Exception:
            pass
        with st.expander("Preview"):
            try:
                if card.path.suffix.lower() == ".csv":
                    st.dataframe(pd.read_csv(card.path).head(10), use_container_width=True, hide_index=True)
                elif card.path.suffix.lower() == ".jsonl":
                    rows = []
                    with card.path.open(encoding="utf-8") as fh:
                        for _, line in zip(range(3), fh):
                            if line.strip():
                                rows.append(json.loads(line))
                    st.json(rows)
            except Exception as exc:
                st.warning(f"Preview failed: {exc}")


def render_exports_page() -> None:
    st.header("Export")
    with st.container(border=True):
        st.subheader("Generate outputs")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate export bundle", use_container_width=True):
                bundle = exporter.export_accounting_grade_bundle(DB_PATH, out_dir="outputs")
                exporter.generate_quickbooks_csv(out_path="outputs/quickbooks_import.csv", db_path=DB_PATH)
                st.success(f"Generated export bundle `{bundle['run_id']}`.")
                st.rerun()
        with c2:
            if st.button("Generate reconciliation queue", use_container_width=True):
                reconciliation.export_reconciliation_reports(DB_PATH, out_dir="outputs/reconciliation")
                st.success("Generated reconciliation queue files.")
                st.rerun()

    cards = [
        ExportCard("Enriched transactions CSV", Path("outputs/accounting_transactions_enriched.csv")),
        ExportCard("GL lines CSV", Path("outputs/gl_lines.csv")),
        ExportCard("Audit trail JSONL", Path("outputs/audit_trail.jsonl")),
        ExportCard("QuickBooks-style CSV", Path("outputs/quickbooks_import.csv")),
    ]
    cols = st.columns(2)
    for i, card in enumerate(cards):
        with cols[i % 2]:
            render_export_card(card)


def render_sidebar(queue_count: int) -> str:
    st.sidebar.title("SourceTax")
    st.sidebar.caption("Finance review console")
    st.sidebar.markdown(f"Exceptions: **{queue_count}**")
    return st.sidebar.radio(
        "Workflow",
        ["Upload / Ingest", "Review Grid", "Exceptions", "Export"],
    )


def main() -> None:
    inject_styles()
    render_header()
    all_df = fetch_all_records_df()
    queue_df = build_review_queue_df(all_df)
    page = render_sidebar(len(queue_df))

    if page == "Upload / Ingest":
        render_ingest_screen(all_df)
    elif page == "Review Grid":
        render_review_grid(all_df)
    elif page == "Exceptions":
        render_exceptions_screen(all_df)
    elif page == "Export":
        render_exports_page()


if __name__ == "__main__":
    main()
