"""Client for interacting with a mem0 MCP server."""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


class Mem0Client:
    """A client for interacting with a mem0 mcp server via langchain-mcp-adapters."""

    def __init__(self, url: str, hass: HomeAssistant) -> None:
        """Initialize the client."""
        self._url = url
        self._hass = hass
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools: dict[str, Any] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect to the MCP server and load tools."""
        if self._connected:
            return

        self._exit_stack = AsyncExitStack()
        try:
            # Connect to SSE endpoint
            # We assume the URL is the full SSE endpoint e.g. http://host:port/sse
            read, write = await self._exit_stack.enter_async_context(sse_client(self._url))
            
            # Create and initialize session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            
            # Load tools into memory using the adapter
            # load_mcp_tools returns a list of LangChain tools
            tools_list = await load_mcp_tools(self._session)
            for tool in tools_list:
                self._tools[tool.name] = tool
            
            self._connected = True
            LOGGER.info("Connected to Mem0 MCP server at %s", self._url)
        except Exception as err:
            LOGGER.error("Failed to connect to Mem0 MCP server: %s", err)
            if self._exit_stack:
                await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            raise

    async def close(self) -> None:
        """Close the client connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._connected = False

    async def save_memory(self, text: str) -> str:
        """Save information to long-term memory."""
        return await self._call_tool("save_memory", {"text": text})

    async def get_all_memories(self) -> str:
        """Get all stored memories."""
        return await self._call_tool("get_all_memories", {})

    async def search_memories(self, query: str, limit: int = 3) -> str:
        """Search memories."""
        return await self._call_tool("search_memories", {"query": query, "limit": limit})

    async def _call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Helper to execute a tool."""
        # Check if we have the tool naturally or possibly prefixed
        # load_mcp_tools usually keeps the original name unless conflicted?
        # Let's try direct first.
        
        target_tool = self._tools.get(tool_name)
        if not target_tool:
            # Fallback simple search if prefixes are involved (e.g. server name?)
            # But here we only have one session, so likely no prefix unless server enforced.
             for name, tool in self._tools.items():
                if name.endswith(tool_name.replace("mem0_", "")):
                    target_tool = tool
                    break
        
        if not target_tool:
            return f"Error: Tool {tool_name} not found. Available: {list(self._tools.keys())}"

        try:
            # LangChain tools are typically run with .invoke() or .ainvoke()
            if hasattr(target_tool, "ainvoke"):
                return await target_tool.ainvoke(args)
            return await asyncio.to_thread(target_tool.invoke, args)
        except Exception as err:
            LOGGER.error("Error calling tool %s: %s", tool_name, err)
            return f"Error: {err}"

    # Shim to allow property access like client.tools.save_memory(text=...)
    @property
    def tools(self) -> _ToolsShim:
        """Shim for tool access."""
        return _ToolsShim(self)


class _ToolsShim:
    def __init__(self, client: Mem0Client):
        self._client = client

    async def save_memory(self, text: str) -> str:
        return await self._client.save_memory(text)

    async def get_all_memories(self) -> str:
        return await self._client.get_all_memories()

    async def search_memories(self, query: str, limit: int = 3) -> str:
        return await self._client.search_memories(query, limit)
