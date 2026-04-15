"""Shared pytest fixtures for paper-monitoring tests."""
import pytest

from src.store.graph_store import GraphStore


@pytest.fixture
def in_memory_store() -> GraphStore:
    """Shared in-memory GraphStore fixture available to all test modules."""
    return GraphStore(":memory:")
