"""
clinicaliq/tools.py  (starter)
-------------------------------
Session 8: replace direct SQLite tool calls with MCP.
query_doctors and query_services are no longer hand-written @tool functions.
They are discovered dynamically from the S07 MCP server using MCPAdapter.

Three TODOs -- one file, zero graph changes:

  TODO 1 -- _load_tools()          implement the async tool-discovery function
  TODO 2 -- mcp_tools / registry   call it, build the registry, bind the LLM
  TODO 3 -- _run_tool()            dispatch a call, bridge async -> sync

Quick self-test (run from s08/starter/):
  python -c "
  import sys; sys.path.insert(0, '.')
  from clinicaliq.tools import _run_tool
  print(_run_tool('query_doctors', {'specialty': 'all'}))
  print(_run_tool('query_services', {'department': 'all'}))
  "
  Expected initially: NotImplementedError -- that confirms the starter is wired.
  After all three TODOs: real clinic data printed from the MCP server.
"""
import asyncio
import sys

from langchain_groq import ChatGroq
from langchain.mcp import MCPAdapter

from .config import (
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_MODEL,
    GROQ_API_KEY,
    MAX_TOKENS,
    MCP_SERVER_PATH,
    MODEL_NAME,
    TEMPERATURE,
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=CLASSIFIER_MODEL,
    temperature=0.0,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)

# ---------------------------------------------------------------------------
# TODO 1 of 3 -- implement _load_tools()
# ---------------------------------------------------------------------------
# _mcp_config tells MCPAdapter how to start the MCP server:
#   - "command": sys.executable   -> use the same venv Python, not system Python
#   - "args":    [str(MCP_SERVER_PATH)] -> path to s07/solution/mcp_server.py
# This is provided. You only need to implement _load_tools() below.
#
# async def _load_tools():
#   Use MCPAdapter(_mcp_config) as an async context manager and return the
#   result of await adapter.list_tools().
#   Shape: async with MCPAdapter(_mcp_config) as adapter:
#              return await adapter.list_tools()
# ---------------------------------------------------------------------------

_mcp_config = {
    "mcpServers": {
        "clinicaliq": {
            "command": sys.executable,          # venv Python -- do not change
            "args":    [str(MCP_SERVER_PATH)],  # s07/solution/mcp_server.py -- do not change
        }
    }
}


async def _load_tools():
    """Discover tools from the MCP server. Called once at import time."""
    # TODO 1: use MCPAdapter(_mcp_config) as async context manager,
    # return await adapter.list_tools()
    raise NotImplementedError("TODO 1: implement _load_tools()")


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- mcp_tools / _tool_registry / llm_with_tools
# ---------------------------------------------------------------------------
# After _load_tools() works, uncomment the three lines below.
#
# mcp_tools      = asyncio.run(_load_tools())
# _tool_registry = {t.name: t for t in mcp_tools}
# llm_with_tools = llm.bind_tools(mcp_tools)
# ---------------------------------------------------------------------------

# mcp_tools      = asyncio.run(_load_tools())      # TODO 2a: uncomment
# _tool_registry = {t.name: t for t in mcp_tools}  # TODO 2b: uncomment
# llm_with_tools = llm.bind_tools(mcp_tools)        # TODO 2c: uncomment
raise NotImplementedError("TODO 2: uncomment the three lines above")


def _extract_text(result) -> str:
    """MCP tool results come back as content blocks: [{"type": "text", "text": "..."}].
    Join only the text blocks into the plain string nodes.py expects."""
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)


# ---------------------------------------------------------------------------
# TODO 3 of 3 -- _invoke_via_mcp() + _run_tool()
# ---------------------------------------------------------------------------
# 1. Implement _invoke_via_mcp(tool_name, tool_args) -> str
#    Open a fresh MCPAdapter(_mcp_config) session per call.
#    Call adapter.list_tools() to build a local registry for this session.
#    Look up tool_name and call .ainvoke(tool_args).
#    Pass the result through _extract_text() and return the string.
#
# 2. Implement _run_tool(tool_name, tool_args) -> str
#    - If tool_name not in _tool_registry: return f"Unknown tool: {tool_name}"
#    - try: return asyncio.run(_invoke_via_mcp(tool_name, tool_args))
#    - except Exception as e: return f"Tool error ({tool_name}): {e}"
# ---------------------------------------------------------------------------

async def _invoke_via_mcp(tool_name: str, tool_args: dict) -> str:
    """Open a fresh MCP session, call tool_name with tool_args, return plain text."""
    # TODO 3a: open MCPAdapter(_mcp_config), list tools, .ainvoke(), _extract_text()
    raise NotImplementedError("TODO 3a: implement _invoke_via_mcp()")


def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call to the MCP server. Called by nodes.py."""
    # TODO 3b: check _tool_registry, asyncio.run(_invoke_via_mcp(...)), handle errors
    raise NotImplementedError("TODO 3b: implement _run_tool()")
