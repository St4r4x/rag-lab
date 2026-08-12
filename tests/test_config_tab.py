# tests/test_config_tab.py
from ui.config_tab import CONFIG_LABELS, format_config_rows


def test_format_config_rows_uses_friendly_labels():
    config = {"llm_model": "ollama:llama3.2:3b", "qdrant_url": "http://localhost:6333"}
    rows = format_config_rows(config)
    assert rows == [
        [CONFIG_LABELS["llm_model"], "ollama:llama3.2:3b"],
        [CONFIG_LABELS["qdrant_url"], "http://localhost:6333"],
    ]


def test_format_config_rows_falls_back_to_key_for_unknown_field():
    rows = format_config_rows({"new_field": "value"})
    assert rows == [["new_field", "value"]]
