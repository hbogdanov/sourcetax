from sourcetax.core import parse_stub


def test_parse_stub_basic():
    s = "Vendor: Coffee Shop; Date: 2026-02-21; Amount: $4.50"
    res = parse_stub(s)
    assert res["vendor"] == "Coffee Shop"
    assert res["date"] == "2026-02-21"
    assert res["amount"] == "4.50"
