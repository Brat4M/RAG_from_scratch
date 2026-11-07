"""Centralized configuration loader using python-dotenv.

Loads environment variables from root `.env` without exposing secrets
or raising errors when optional keys are absent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Find project root (directory containing pyproject.toml or .env)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables once at module import
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Read-only interface to project settings."""

    @property
    def project_name(self) -> str:
        return os.getenv("PROJECT_NAME", "rag-engineering-academy")

    @property
    def llm_provider(self) -> str:
        return os.getenv("LLM_PROVIDER", "groq")

    @property
    def groq_api_key(self) -> Optional[str]:
        return os.getenv("GROQ_API_KEY") or None

    @property
    def chroma_persist_directory(self) -> Path:
        raw = os.getenv("CHROMA_PERSIST_DIRECTORY", "./storage/chroma")
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @property
    def langsmith_tracing(self) -> bool:
        return os.getenv("LANGSMITH_TRACING", "false").strip().lower() in ("true", "1", "yes")

    @property
    def langsmith_api_key(self) -> Optional[str]:
        return os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or None

    @property
    def langsmith_project(self) -> str:
        return os.getenv("LANGSMITH_PROJECT", "rag-engineering-academy")

    def require_groq_key(self) -> str:
        """Validate and return GROQ_API_KEY, raising ValueError if missing."""
        key = self.groq_api_key
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not configured in .env. "
                "Please add your Groq API key to .env to execute optional live LLM cells."
            )
        return key


settings = Settings()
