"""Tests for common configuration and settings loading."""

import pytest
from common.config import settings


def test_settings_defaults():
    assert settings.project_name in ("rag-engineering-academy", "RAG_from_scratch")
    assert settings.llm_provider == "groq"
    assert settings.chroma_persist_directory is not None


def test_settings_require_groq_key_when_empty(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Reload settings behavior under monkeypatched env
    from common.config import Settings
    custom_settings = Settings()
    if not custom_settings.groq_api_key:
        with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
            custom_settings.require_groq_key()
