"""
SourceTax Review UI (Streamlit).

Goals:
- Fast skimmable dashboard for demos
- Human-in-the-loop review queue
- Matching review with score breakdown
- Export visibility (accounting outputs)
- Gold set progress / labeling workflow controls

This app avoids brittle JSON parsing by using safe decoding for DB fields.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sourcetax import categorization, exporter, matching, reconciliation, storage


DB_PATH = "data/store.db"
GOLD_PATH = "data/gold/gold_transactions.jsonl"
GOLD_TARGET = 200


st.set_page_config(page_title="SourceTax Review", page_icon=None, layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .main .block-container {max-width: 1150px; padding-top: 1.2rem; padding-bottom: 2rem;}
          .app-title {font-size: 1.8rem; font-weight: 800; margin-bottom: 0.15rem;}
          .app-sub {color: #5b6773; margin-bottom: 0.9rem;}
          .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}

          /* Small UI polish */
          div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,0.12); border-radius: 12px; padding: 0.75rem;}
          .stButton>button {border-radius: 10px; padding: 0.55rem 0.8rem;}
          div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden; border: 1px solid rgba(49,51,63,0.12);}
          div[data-testid="stExpander"] {border-radius: 12px; border: 1px solid rgba(49,51,63,0.10);}

          /* Badges used in detail panel */
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
    try:
        v = float(value)
    except Exception:
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
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
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


def query_rows(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def parse_record_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    rec["raw_payload"] = safe_json_loads(rec.get("raw_payload"), {})
    rec["evidence_keys"] = safe_json_loads(rec.get("evidence_keys"), [])
    rec["tags"] = safe_json_loads(rec.get("tags"), [])
    return rec


def fetch_record(record_id: str) -> Optional[Dict[str, Any]]:
    rows = query_rows("SELECT * FROM canonical_records WHERE id = ?", (record_id,))
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
    needs_review = recon.get("low_confidence_queue_size", 0) + recon.get("conflicts_queue_size", 0) + max(receipts - matched_receipts, 0)
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


def build_review_queue_df(all_df: pd.DataFrame) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()
    low_conf = {r["id"]: r for r in reconciliation.low_confidence_categorizations(DB_PATH)}
    conflicts = {r["id"]: r for r in reconciliation.conflicts_queue(DB_PATH)}
    unmatched_receipts = {r["id"]: r for r in reconciliation.unmatched_receipts(DB_PATH)}
    unmatched_bank = {r["id"]: r for r in reconciliation.unmatched_bank_transactions(DB_PATH)}
    rows = []
    for _, row in all_df.iterrows():
        rid = row.get("id")
        if not rid:
            continue
        if rid not in low_conf and rid not in conflicts and rid not in unmatched_receipts and rid not in unmatched_bank:
            continue
        raw_payload = row.get("raw_payload") if isinstance(row.get("raw_payload"), dict) else {}
        issue = "low_confidence"
        if rid in conflicts:
            issue = "conflict"
        elif rid in unmatched_receipts:
            issue = "unmatched_receipt"
        elif rid in unmatched_bank:
            issue = "unmatched_bank_txn"
        rows.append(
            {
                "id": rid,
                "transaction_date": row.get("transaction_date"),
                "merchant_raw": row.get("merchant_raw"),
                "merchant_norm": row.get("merchant_norm"),
                "amount": row.get("amount"),
                "source": row.get("source"),
                "predicted_category": row.get("category_pred"),
                "final_category": row.get("category_final"),
                "effective_category": row.get("category_final") or row.get("category_pred") or "Uncategorized",
                "confidence": row.get("confidence"),
                "issue_type": issue,
                "rules_pred": conflicts.get(rid, {}).get("rules_pred"),
                "ml_pred": conflicts.get(rid, {}).get("ml_pred") or (raw_payload.get("ml_prediction") if isinstance(raw_payload, dict) else None),
                "label_source": label_source_for_record({"category_final": row.get("category_final"), "raw_payload": raw_payload}),
            }
        )
    q = pd.DataFrame(rows)
    if not q.empty:
        q = q.sort_values(by=["transaction_date", "confidence"], ascending=[False, True], na_position="last")
    return q


def category_options() -> List[str]:
    return [
        "Uncategorized",
        "Meals & Entertainment",
        "Travel",
        "Supplies",
        "Office Equipment",
        "Utilities",
        "Rent",
        "Insurance",
        "Taxes & Licenses",
        "Other",
        "Income",
    ]


def render_header() -> None:
    meta = get_run_metadata()
    st.markdown("<div class='app-title'>SourceTax</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-sub'>Transaction classification • matching • reconciliation • accounting exports</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Run metadata"):
        st.write(f"DB: `{meta['db']}`")
        st.write(f"Commit: `{meta['commit']}`")
        st.write(f"Timestamp: `{meta['timestamp']}`")

def run_pipeline_actions() -> None:
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        if st.button("Auto-match receipts", width="stretch"):
            count = matching.match_all_receipts(DB_PATH)
            st.success(f"Matched {count} receipts.")
            st.rerun()
    with c2:
        if st.button("Auto-categorize", width="stretch"):
            count = categorization.categorize_all_records(DB_PATH)
            st.success(f"Categorized {count} records.")
            st.rerun()
    with c3:
        if st.button("Export gold labels", width="stretch"):
            result = exporter.export_gold_transactions_jsonl(DB_PATH, GOLD_PATH, append=True)
            st.success(f"Exported {result['exported']} labels. Gold total: {result['total_after']}")


def render_dashboard(all_df: pd.DataFrame) -> None:
    st.header("Dashboard")
    stats = dashboard_stats(all_df)
    recon_summary = reconciliation.summary_metrics(DB_PATH)

    with st.container(border=True):
        run_pipeline_actions()

    cols = st.columns(6)
    cols[0].metric("Total transactions", stats["total_records"])
    cols[1].metric("Receipts", stats["receipts"])
    cols[2].metric("Match rate", fmt_pct(stats["match_rate"]))
    cols[3].metric("Categorized", fmt_pct(stats["categorization_coverage"]))
    cols[4].metric("Avg confidence", fmt_pct(stats["avg_confidence"]) if stats["avg_confidence"] is not None else "-")
    cols[5].metric("Needs review", stats["needs_review"])

    c1, c2 = st.columns([1.25, 1.0])
    with c1:
        with st.container(border=True):
            st.subheader("Spend by Category")
            if all_df.empty:
                st.info("No records loaded.")
            else:
                df = all_df.copy()
                df["effective_category"] = df["category_final"].fillna("").replace("", pd.NA).fillna(df["category_pred"]).fillna("Uncategorized")
                df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
                expenses = df[df["direction"] == "expense"]
                if expenses.empty:
                    st.info("No expense rows yet.")
                else:
                    spend = (
                        expenses.groupby("effective_category")["amount_num"]
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"effective_category": "Category", "amount_num": "Spend"})
                    )
                    top = spend.head(10).copy()
                    st.dataframe(
                        top,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Spend": st.column_config.NumberColumn("Spend", format="$%.2f"),
                        },
                        height=300,
                    )
                    st.bar_chart(top.set_index("Category")["Spend"])
    with c2:
        with st.container(border=True):
            st.subheader("Review Queue Breakdown")
            breakdown = pd.DataFrame(
                [
                    ("Unmatched Receipts", len(reconciliation.unmatched_receipts(DB_PATH))),
                    ("Unmatched Bank Txns", len(reconciliation.unmatched_bank_transactions(DB_PATH))),
                    ("Low Confidence", int(recon_summary.get("low_confidence_queue_size", 0) or 0)),
                    ("Conflicts", int(recon_summary.get("conflicts_queue_size", 0) or 0)),
                ],
                columns=["Issue", "Count"],
            ).sort_values("Count", ascending=False)
            st.dataframe(breakdown, hide_index=True, use_container_width=True, height=220)
            st.bar_chart(breakdown.set_index("Issue")["Count"])
            st.markdown("**Next actions**")
            st.write(f"- Review **{int(breakdown[breakdown['Issue']=='Low Confidence']['Count'].iloc[0]) if (breakdown['Issue']=='Low Confidence').any() else 0}** low-confidence items")
            st.write(f"- Resolve **{int(breakdown[breakdown['Issue']=='Conflicts']['Count'].iloc[0]) if (breakdown['Issue']=='Conflicts').any() else 0}** conflicts")
            st.write("- Generate accounting exports")

def render_record_detail_panel(record: Dict[str, Any], queue_row: Optional[Dict[str, Any]] = None) -> None:
    rec = parse_record_fields(record)
    c1, c2 = st.columns([1.05, 1.2])
    with c1:
        with st.container(border=True):
            st.subheader("Transaction")
            st.write(f"Date: `{rec.get('transaction_date') or '-'}`")
            st.write(f"Merchant: **{rec.get('merchant_raw') or '-'}**")
            st.write(f"Normalized: `{rec.get('merchant_norm') or '-'}`")
            st.write(f"Amount: {fmt_money(rec.get('amount'))}")
            st.write(f"Source: `{rec.get('source') or '-'}`")
            st.markdown(confidence_badge_html(rec.get('confidence')), unsafe_allow_html=True)
            current = rec.get('category_final') or rec.get('category_pred') or 'Uncategorized'
            st.write(f"Category: **{current}**")
            st.write(f"Label source: `{label_source_for_record(rec)}`")
            if rec.get('matched_transaction_id'):
                st.write(f"Matched: `{rec.get('matched_transaction_id')}` ({fmt_pct(rec.get('match_score'))})")
    with c2:
        with st.container(border=True):
            st.subheader("Evidence / Details")
            for ev in (rec.get("evidence_keys") or [])[:10]:
                st.write(f"- `{ev}`")
            raw_payload = rec.get("raw_payload") or {}
            if isinstance(raw_payload, dict):
                ocr = raw_payload.get("ocr_text")
                if ocr:
                    st.text_area("Receipt excerpt", str(ocr)[:800], height=160, disabled=True)
                if queue_row and queue_row.get("issue_type") == "conflict":
                    st.write(f"Rules prediction: `{queue_row.get('rules_pred') or '-'}`")
                    st.write(f"ML prediction: `{queue_row.get('ml_pred') or '-'}`")
                top3 = raw_payload.get("top3_predictions")
                if isinstance(top3, list) and top3:
                    st.write("Top predictions:")
                    for item in top3[:3]:
                        if isinstance(item, dict):
                            st.write(f"- {item.get('category', '?')}: {fmt_pct(item.get('prob'))}")

    with st.container(border=True):
        st.subheader("Review Action")
        options = category_options()
        current = rec.get("category_final") or rec.get("category_pred") or "Uncategorized"
        idx = options.index(current) if current in options else 0
        x1, x2, x3 = st.columns([2.2, 1.1, 1.1])
        with x1:
            override = st.selectbox("Override category", options, index=idx, key=f"override_{rec.get('id')}")
        with x2:
            if st.button("Approve", key=f"approve_{rec.get('id')}", width="stretch"):
                categorization.save_category_override(rec["id"], current, DB_PATH)
                st.success(f"Saved category: {current}")
                st.rerun()
        with x3:
            if st.button("Save", key=f"save_{rec.get('id')}", width="stretch"):
                categorization.save_category_override(rec["id"], override, DB_PATH)
                st.success(f"Saved category: {override}")
                st.rerun()


def render_review_queue(all_df: pd.DataFrame) -> None:
    st.header("Review Queue")
    q = build_review_queue_df(all_df)
    if q.empty:
        st.success("No review queue items.")
        return

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.5, 2.2])
        issue_options = ["all"] + sorted(q["issue_type"].dropna().unique().tolist())
        cat_options = ["all"] + sorted(q["effective_category"].fillna("Uncategorized").unique().tolist())
        issue_filter = f1.selectbox("Issue type", issue_options)
        category_filter = f2.selectbox("Category", cat_options)
        conf_range = f3.slider("Confidence", 0.0, 1.0, (0.0, 1.0), step=0.05)
        merchant_search = f4.text_input("Merchant search", placeholder="Starbucks, Uber, Amazon…")

    qf = q.copy()
    qf["confidence_num"] = pd.to_numeric(qf["confidence"], errors="coerce")
    if issue_filter != "all":
        qf = qf[qf["issue_type"] == issue_filter]
    if category_filter != "all":
        qf = qf[qf["effective_category"] == category_filter]
    qf = qf[(qf["confidence_num"].fillna(-1).between(conf_range[0], conf_range[1])) | (qf["confidence_num"].isna())]
    if merchant_search.strip():
        term = merchant_search.strip().lower()
        qf = qf[
            qf["merchant_raw"].fillna("").str.lower().str.contains(term)
            | qf["merchant_norm"].fillna("").str.lower().str.contains(term)
        ]

    if qf.empty:
        st.info("No items match the current filters.")
        return

    def _short_id(x: Any) -> str:
        s = str(x or "")
        return s[:8] if s else ""

    left, right = st.columns([1.35, 1.0])
    with left:
        with st.container(border=True):
            st.subheader(f"Queue ({len(qf)} items)")

            display = qf[["id", "transaction_date", "merchant_raw", "amount", "effective_category", "confidence_num", "issue_type"]].copy()
            display["id"] = display["id"].map(_short_id)
            display.rename(
                columns={
                    "id": "ID",
                    "transaction_date": "Date",
                    "merchant_raw": "Merchant",
                    "amount": "Amount",
                    "effective_category": "Category",
                    "confidence_num": "Confidence",
                    "issue_type": "Issue",
                },
                inplace=True,
            )

            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                height=420,
                column_config={
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "Date": st.column_config.DateColumn("Date"),
                    "Issue": st.column_config.TextColumn("Issue", help="Why this row needs review"),
                },
            )

            # Selection (human-friendly labels instead of UUID soup)
            labels = (
                qf["transaction_date"].fillna("").astype(str)
                + " • "
                + qf["merchant_raw"].fillna("Unknown merchant").astype(str).str.slice(0, 40)
                + " • "
                + qf["amount"].fillna("").astype(str)
            ).tolist()
            idx = st.radio(
                "Open item",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                key="review_select_idx",
            )
            selected_id = qf.iloc[idx]["id"]

    with right:
        if selected_id:
            rec = fetch_record(selected_id)
            row = qf[qf["id"] == selected_id].iloc[0].to_dict()
            if rec:
                render_record_detail_panel(rec, queue_row=row)

def match_candidates_for_receipt(receipt: Dict[str, Any], top_k: int = 5) -> pd.DataFrame:
    candidates = query_rows(
        """
        SELECT id, source, transaction_date, merchant_raw, amount
        FROM canonical_records
        WHERE source IN ('bank','toast','quickbooks')
        """
    )
    rows = []
    for cand in candidates:
        d = matching.date_closeness_score(receipt.get("transaction_date"), cand.get("transaction_date"), max_days=3)
        a = matching.amount_closeness_score(receipt.get("amount"), cand.get("amount"), tolerance=10.0)
        m = matching.merchant_similarity_score(receipt.get("merchant_raw"), cand.get("merchant_raw"), min_ratio=0.7)
        score = d * 0.3 + a * 0.5 + m * 0.2
        rows.append({
            "candidate_id": cand.get("id"),
            "source": cand.get("source"),
            "transaction_date": cand.get("transaction_date"),
            "merchant_raw": cand.get("merchant_raw"),
            "amount": cand.get("amount"),
            "score": score,
            "amount_score": a,
            "date_score": d,
            "merchant_score": m,
            "strength": "strong" if score >= 0.85 else ("medium" if score >= 0.65 else "weak"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).head(top_k)


def link_receipt(receipt_id: str, transaction_id: str, score: float) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE canonical_records SET matched_transaction_id = ?, match_score = ? WHERE id = ?", (transaction_id, float(score), receipt_id))
    conn.commit()
    conn.close()


def render_matching_page() -> None:
    st.header("Matching")
    receipts = pd.DataFrame(reconciliation.unmatched_receipts(DB_PATH))
    if receipts.empty:
        st.success("No unmatched receipts.")
        return

    left, right = st.columns([1.0, 1.3])
    with left:
        with st.container(border=True):
            st.subheader("Unmatched Receipts")
            st.dataframe(receipts[["id", "transaction_date", "merchant_raw", "amount"]], width="stretch", hide_index=True)
            selected_receipt_id = st.selectbox("Select receipt", receipts["id"].tolist(), key="matching_receipt")
    with right:
        receipt = fetch_record(selected_receipt_id) if selected_receipt_id else None
        if not receipt:
            return
        with st.container(border=True):
            st.subheader("Suggested Matches")
            st.write(f"Receipt: **{receipt.get('merchant_raw', '-') }** | {fmt_money(receipt.get('amount'))} | `{receipt.get('transaction_date', '-')}`")
            cand_df = match_candidates_for_receipt(receipt, top_k=5)
            if cand_df.empty:
                st.info("No candidates found.")
                return
            st.dataframe(cand_df, width="stretch", hide_index=True)
            cand_ids = [str(x) if x is not None else "(missing-id)" for x in cand_df["candidate_id"].tolist()]
            selected = st.selectbox("Candidate to link", cand_ids)
            sel_row = cand_df.iloc[cand_ids.index(selected)]
            st.write("Score breakdown")
            st.progress(float(sel_row["score"]))
            b1, b2, b3 = st.columns(3)
            b1.metric("Amount score", fmt_pct(sel_row["amount_score"]))
            b2.metric("Date score", fmt_pct(sel_row["date_score"]))
            b3.metric("Merchant score", fmt_pct(sel_row["merchant_score"]))
            st.write(f"Strength: `{sel_row['strength']}`")
            if selected != "(missing-id)" and st.button("Link selected match", width="stretch"):
                link_receipt(receipt["id"], selected, float(sel_row["score"]))
                st.success(f"Linked receipt {receipt['id']} to transaction {selected}")
                st.rerun()


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
        st.write(f"Updated: {datetime.fromtimestamp(card.path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            data = card.path.read_bytes()
            st.download_button("Download", data=data, file_name=card.path.name, width="stretch")
        except Exception:
            pass
        with st.expander("Preview"):
            try:
                if card.path.suffix.lower() == ".csv":
                    st.dataframe(pd.read_csv(card.path).head(10), width="stretch", hide_index=True)
                elif card.path.suffix.lower() == ".jsonl":
                    rows = []
                    with card.path.open(encoding="utf-8") as fh:
                        for _, line in zip(range(3), fh):
                            if line.strip():
                                rows.append(json.loads(line))
                    st.json(rows)
            except Exception as e:
                st.warning(f"Preview failed: {e}")


def render_exports_page() -> None:
    st.header("Exports")
    with st.container(border=True):
        st.subheader("Generate outputs")
        x1, x2, x3 = st.columns(3)
        with x1:
            if st.button("Generate basic exports", width="stretch"):
                qb = exporter.generate_quickbooks_csv(db_path=DB_PATH)
                totals, counts = exporter.compute_schedule_c_totals(DB_PATH)
                exporter.write_schedule_c_csv(totals, counts)
                exporter.export_audit_pack(DB_PATH)
                st.success(f"Generated {qb} and related basic exports.")
                st.rerun()
        with x2:
            if st.button("Generate accounting-grade bundle", width="stretch"):
                result = exporter.export_accounting_grade_bundle(DB_PATH, out_dir="outputs")
                reconciliation.export_reconciliation_reports(DB_PATH, out_dir="outputs/reconciliation")
                st.success(f"Generated accounting-grade bundle (run_id={result['run_id']}).")
                st.rerun()
        with x3:
            if st.button("Generate reconciliation reports", width="stretch"):
                reconciliation.export_reconciliation_reports(DB_PATH, out_dir="outputs/reconciliation")
                st.success("Generated reconciliation queue CSVs and summary metrics.")
                st.rerun()

    cards = [
        ExportCard("QuickBooks import CSV", Path("outputs/quickbooks_import.csv")),
        ExportCard("Schedule C totals CSV", Path("outputs/schedule_c_totals.csv")),
        ExportCard("Audit pack CSV", Path("outputs/audit_pack.csv")),
        ExportCard("GL lines CSV", Path("outputs/gl_lines.csv")),
        ExportCard("Audit trail JSONL", Path("outputs/audit_trail.jsonl")),
        ExportCard("Needs review CSV (benchmark)", Path("reports/phase3_eval_assets/needs_review.csv")),
        ExportCard("Low confidence queue CSV", Path("outputs/reconciliation/low_confidence_categorizations.csv")),
    ]
    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3]:
            render_export_card(card)


def render_gold_page() -> None:
    st.header("Gold Set (Labeling)")
    gold_count = exporter.count_gold_records(GOLD_PATH)
    st.progress(min(gold_count / GOLD_TARGET, 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Gold labels", gold_count)
    c2.metric("Target", GOLD_TARGET)
    c3.metric("Remaining", max(GOLD_TARGET - gold_count, 0))

    with st.container(border=True):
        x1, x2 = st.columns(2)
        with x1:
            if st.button("Export reviewed labels to gold set", width="stretch"):
                result = exporter.export_gold_transactions_jsonl(DB_PATH, GOLD_PATH, append=True)
                st.success(f"Exported {result['exported']} labels. Gold total: {result['total_after']}")
                st.rerun()
        with x2:
            st.info("For batch labeling: run the labeling UI in a separate terminal:\n\nstreamlit run tools/labeling_ui.py -- --input data/ml/selection_for_labeling.csv")

    p = Path(GOLD_PATH)
    with st.container(border=True):
        st.subheader("Category Distribution")
        if not p.exists():
            st.info("Gold file not found.")
            return
        rows = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not rows:
            st.info("Gold file is empty.")
            return
        df = pd.DataFrame(rows)
        cat_col = "category_final" if "category_final" in df.columns else ("category" if "category" in df.columns else None)
        if not cat_col:
            st.info("No category field found in gold file.")
            return
        counts = df[cat_col].fillna("Unlabeled").value_counts()
        st.bar_chart(counts)
        st.dataframe(counts.rename_axis("Category").reset_index(name="Count"), width="stretch", hide_index=True)


def render_sidebar(queue_count: int) -> str:
    st.sidebar.title("SourceTax")
    st.sidebar.caption("Review and export console")
    st.sidebar.markdown(f"Queue items: **{queue_count}**")
    return st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Review Queue", "Matching", "Exports", "Gold Set (Labeling)"],
    )


def main() -> None:
    inject_styles()
    render_header()
    all_df = fetch_all_records_df()
    queue_df = build_review_queue_df(all_df)
    page = render_sidebar(len(queue_df))

    if page == "Dashboard":
        render_dashboard(all_df)
    elif page == "Review Queue":
        render_review_queue(all_df)
    elif page == "Matching":
        render_matching_page()
    elif page == "Exports":
        render_exports_page()
    elif page == "Gold Set (Labeling)":
        render_gold_page()


if __name__ == "__main__":
    main()
