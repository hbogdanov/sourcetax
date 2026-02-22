from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json


@dataclass
class CanonicalRecord:
    # Identifiers
    id: Optional[str]  # stable UUID
    record_id: Optional[str]  # alias for id (migration compatibility)
    source: Optional[str]  # bank|receipt|toast|quickbooks|manual
    source_record_id: Optional[str]  # row id / filename + index

    # Temporal
    transaction_date: Optional[str]  # ISO 8601

    # Merchant
    merchant_raw: Optional[str]  # as-written from source
    merchant_norm: Optional[str]  # normalized: lowercase, punctuation stripped
    merchant_name: Optional[str]  # alias for backward compat (use merchant_raw)

    # Amount & currency
    amount: Optional[float]  # always positive
    currency: Optional[str]  # ISO 4217 (default "USD")
    direction: Optional[str]  # "expense" | "income" (no sign ambiguity)
    payment_method: Optional[str]

    # Categorization
    category_pred: Optional[str]  # predicted category (from rules/ML)
    category_final: Optional[str]  # user override (if present, use for exports)
    confidence: Optional[float]  # single score 0-1 (high=0.95, medium=0.6, low=0.3)

    # Matching & Evidence
    matched_transaction_id: Optional[str]  # reference to linked bank transaction
    match_score: Optional[float]  # 0-1, confidence of match
    evidence_keys: Optional[List[str]]  # ["ocr_text", "receipt_file.jpg", "bank_memo"]

    # Raw data & metadata
    raw_payload: Optional[Dict[str, Any]]  # full extracted/original data
    tags: Optional[List[str]]

    def to_row(self) -> Dict[str, Any]:
        return {
            "id": self.id or self.record_id,
            "source": self.source,
            "source_record_id": self.source_record_id,
            "transaction_date": self.transaction_date,
            "merchant_raw": self.merchant_raw or self.merchant_name,
            "merchant_norm": self.merchant_norm,
            "amount": self.amount,
            "currency": self.currency,
            "direction": self.direction,
            "payment_method": self.payment_method,
            "category_pred": self.category_pred,
            "category_final": self.category_final,
            "confidence": self.confidence,
            "matched_transaction_id": self.matched_transaction_id,
            "match_score": self.match_score,
            "evidence_keys": json.dumps(self.evidence_keys or []),
            "raw_payload": json.dumps(self.raw_payload or {}),
            "tags": json.dumps(self.tags or []),
        }

    @staticmethod
    def from_normalized(d: Dict[str, Any]) -> "CanonicalRecord":
        return CanonicalRecord(
            id=d.get("id"),
            record_id=d.get("record_id"),
            source=d.get("source"),
            source_record_id=d.get("source_record_id"),
            transaction_date=d.get("transaction_date"),
            merchant_raw=d.get("merchant_raw"),
            merchant_norm=d.get("merchant_norm"),
            merchant_name=d.get("merchant_name"),  # backward compat
            amount=d.get("amount"),
            currency=d.get("currency"),
            direction=d.get("direction"),
            payment_method=d.get("payment_method"),
            category_pred=d.get("category_pred"),
            category_final=d.get("category_final"),
            confidence=d.get("confidence"),
            matched_transaction_id=d.get("matched_transaction_id"),
            match_score=d.get("match_score"),
            evidence_keys=d.get("evidence_keys"),
            raw_payload=d.get("raw_payload"),
            tags=d.get("tags"),
        )
