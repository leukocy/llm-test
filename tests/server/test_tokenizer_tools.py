"""Tokenizer setup tools never download or load unregistered remote code."""

from __future__ import annotations

import pytest

from server.tokenizer_tools import count_text, tokenizer_catalog


def test_catalog_maps_model_and_lists_registered_local_status():
    catalog = tokenizer_catalog("DeepSeek-V3.2")
    assert catalog["matched_name"] == "DeepSeek-V3.2"
    assert any(item["name"] == "DeepSeek-V3.2" for item in catalog["items"])


def test_count_modes_are_explicit_about_units_and_reject_arbitrary_path(monkeypatch):
    assert count_text("汉字 A", "characters", None) == {
        "count": 4,
        "unit": "characters",
        "method": "Unicode code points",
    }

    class FakeEncoding:
        def encode(self, text):
            assert text == "Hello world"
            return [1, 2]

    monkeypatch.setattr("tiktoken.get_encoding", lambda name: FakeEncoding())
    reference = count_text("Hello world", "tiktoken", None)
    assert reference["count"] == 2
    assert "reference" in reference["method"]
    with pytest.raises(ValueError, match="registered"):
        count_text("hello", "local", "../../private")
