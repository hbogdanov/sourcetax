from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json


@dataclass
class CanonicalRecord:
    id: Optional[str]
    merchant_name: Optional[str]
    transaction_date: Optional[str]
    amount: Optional[float]
    currency: Optional[str]
    payment_method: Optional[str]
    source: Optional[str]
    raw_payload: Optional[Dict[str, Any]]
    confidence: Optional[Dict[str, float]]
    tags: Optional[List[str]]

    def to_row(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'merchant_name': self.merchant_name,
            'transaction_date': self.transaction_date,
            'amount': self.amount,
            'currency': self.currency,
            'payment_method': self.payment_method,
            'source': self.source,
            'raw_payload': json.dumps(self.raw_payload or {}),
            'confidence': json.dumps(self.confidence or {}),
            'tags': json.dumps(self.tags or [])
        }

    @staticmethod
    def from_normalized(d: Dict[str, Any]) -> 'CanonicalRecord':
        return CanonicalRecord(
            id=d.get('id'),
            merchant_name=d.get('merchant_name'),
            transaction_date=d.get('transaction_date'),
            amount=d.get('amount'),
            currency=d.get('currency'),
            payment_method=d.get('payment_method'),
            source=d.get('source'),
            raw_payload=d.get('raw_payload'),
            confidence=d.get('confidence'),
            tags=d.get('tags')
        )
