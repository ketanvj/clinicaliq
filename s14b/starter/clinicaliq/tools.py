"""
clinicaliq/tools.py
-------------------
LLM clients and MCP-backed tool loading for ClinicalIQ.

S14b: LlamaGuard 3 8B backend — conditional instantiation.
  LLAMAGUARD_BACKEND=ollama    → ChatOllama (local, free)
  LLAMAGUARD_BACKEND=together  → ChatOpenAI pointed at Together AI (cloud)
"""
import asyncio
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, field_validator

from .config import (
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_MODEL,
    GROQ_API_KEY,
    LLAMAGUARD_BACKEND,
    LLAMAGUARD_MAX_TOKENS,
    LLAMAGUARD_MODEL_OLLAMA,
    LLAMAGUARD_MODEL_TOGETHER,
    MAX_TOKENS,
    MCP_SERVER_PATH,
    MODEL_NAME,
    TEMPERATURE,
    TOGETHER_API_KEY,
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
# S14b: LlamaGuard 3 8B — conditional backend instantiation
#
# Ollama (default): runs locally, free, no API key needed.
#   Uses ChatOllama with num_predict instead of max_tokens.
#
# Together AI: cloud-hosted, same model, requires TOGETHER_API_KEY.
#   Uses ChatOpenAI pointed at Together AI's OpenAI-compatible REST API.
#   This pattern (ChatOpenAI + custom base_url) works for any provider
#   that exposes an OpenAI-compatible endpoint.
# ---------------------------------------------------------------------------
if LLAMAGUARD_BACKEND == "together":
    from langchain_openai import ChatOpenAI
    llamaguard_llm = ChatOpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
        model=LLAMAGUARD_MODEL_TOGETHER,
        temperature=0.0,
        max_tokens=LLAMAGUARD_MAX_TOKENS,
    )
else:  # ollama (default)
    from langchain_ollama import ChatOllama
    llamaguard_llm = ChatOllama(
        model=LLAMAGUARD_MODEL_OLLAMA,
        temperature=0.0,
        num_predict=LLAMAGUARD_MAX_TOKENS,
    )

_mcp_client = MultiServerMCPClient({
    "clinicaliq": {
        "transport": "stdio",
        "command":   sys.executable,
        "args":      [str(MCP_SERVER_PATH)],
    }
})

mcp_tools      = asyncio.run(_mcp_client.get_tools())
_tool_registry = {t.name: t for t in mcp_tools}

llm_with_tools = llm.bind_tools(mcp_tools)


class ToolResponse(BaseModel):
    content: str
    is_error: bool = False

    @field_validator("content")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tool response is empty")
        return v.strip()


def _extract_text(result) -> str:
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)


def _run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in _tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        raw  = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
        text = _extract_text(raw)
        try:
            return ToolResponse(content=text).content
        except Exception as ve:
            print(f"[ClinicalIQ] Tool response validation failed ({tool_name}): {ve}")
            return f"Tool returned invalid response: {tool_name}"
    except Exception as e:
        return f"Error executing tool {tool_name}: {e}"
