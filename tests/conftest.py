"""Shared pytest fixtures.

Forces in-memory Qdrant for the whole suite so tests stay fast and isolated,
regardless of any ``QDRANT_PATH``/``QDRANT_URL`` set in a developer's ``.env``.
"""

import pytest

from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def _force_in_memory_qdrant(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_PATH", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
