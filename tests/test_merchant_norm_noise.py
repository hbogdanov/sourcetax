from sourcetax.normalization import generate_noisy_merchant_raw, normalize_merchant_name
from sourcetax.models.merchant_normalizer import load_merchant_aliases


def test_merchant_normalization_handles_intermediaries_and_ids():
    assert normalize_merchant_name("SQ *STARBUCKS COFFEE 1234 SF CA") == "starbucks"
    assert normalize_merchant_name("PAYPAL *SPOTIFY PREMIUM 998877") == "spotify"
    assert normalize_merchant_name("AMZN MKTP US*2L3AB STORE 1234 WA") == "amazon"
    assert normalize_merchant_name("TST*JOES PIZZA TERMINAL 4455 NY NY") in {
        "joes pizza",
        "joe s pizza",
    }


def test_noise_generator_produces_variants():
    variants = generate_noisy_merchant_raw("Starbucks", n=6, seed=42)
    assert len(variants) >= 3
    assert len(set(variants)) == len(variants)
    for v in variants:
        assert v == v.upper()
        assert "STARBUCKS" in v or len(v) >= 4


def test_aliases_are_loaded_from_data_file():
    aliases = load_merchant_aliases()
    assert aliases.get("STARBUCKS") == "Starbucks"
    assert aliases.get("AMZN MKTP") == "Amazon"
