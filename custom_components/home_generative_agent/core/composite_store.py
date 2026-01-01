"""A composite store that delegates to mem0 or postgres based on namespace."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence, TypedDict

from langgraph.store.base import BaseStore

try:
    from langgraph.store.postgres import StoredSearchHit as SearchResult
except ImportError:

    class SearchResult(TypedDict):
        """A search result."""

        key: str
        """The key of the search result."""
        value: dict[str, Any]
        """The value of the search result."""
        score: float | None
        """The score of the search result."""


from .mem0_client import Mem0Client


class CompositeStore(BaseStore):
    """A composite store that delegates to mem0 or postgres based on namespace."""

    def __init__(self, mem0_store: Mem0Client | None, postgres_store: BaseStore):
        """Initialize the composite store."""
        self.mem0_store = mem0_store
        self.postgres_store = postgres_store

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> dict[str, Any] | None:
        """Get a value from the appropriate store."""
        if self.mem0_store and namespace and namespace[-1] == "memories":
            return None  # mem0 does not support get by key
        return await self.postgres_store.aget(namespace, key)

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Put a value into the appropriate store."""
        if self.mem0_store and namespace and namespace[-1] == "memories":
            content = value.get("content", "")
            context = value.get("context", "")
            await self.mem0_store.save_memory(
                f"Content: {content}\nContext: {context}"
            )
        else:
            await self.postgres_store.aput(namespace, key, value)

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int | None = None,
    ) -> Sequence[SearchResult]:
        """Search in the appropriate store."""
        if self.mem0_store and namespace and namespace[-1] == "memories":
            if not query:
                return []
            results = await self.mem0_store.search_memories(query, limit=limit)
            return [SearchResult(key="", value={"content": r}, score=0.0) for r in results]
        return await self.postgres_store.asearch(namespace, query, limit)

    def acreate(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *, 
        must_not_exist: bool = True,
    ) -> AsyncIterator[str]:
        """Create a value in the appropriate store."""
        if self.mem0_store and namespace and namespace[-1] == "memories":

            async def _acreate() -> AsyncIterator[str]:
                await self.aput(namespace, key, value)
                yield key

            return _acreate()

        return self.postgres_store.acreate(
            namespace, key, value, must_not_exist=must_not_exist
        )
