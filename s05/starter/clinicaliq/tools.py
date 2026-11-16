"""
clinicaliq/tools.py
-------------------
Tool definitions and LLM setup for ClinicalIQ.

Session 5 adds two @tool functions that give the LLM live access to the
Apollo Health Clinic database. Instead of relying only on RAG policy chunks
(Session 4), the LLM can now call query_doctors() and query_services() to
get exact, up-to-date data.

New this session:
  query_doctors()  -- fetches doctor availability and fees from the DB
  query_services() -- fetches diagnostic service prices and durations
  TOOLS            -- list of available tools (used by both bind_tools and _run_tool)
  llm_with_tools   -- the respond LLM with tools bound
  _run_tool()      -- helper that executes whichever tool the LLM requested

Carries forward from Session 4:
  llm             -- still used internally; llm_with_tools wraps it
  classifier_llm  -- unchanged: classifier does not use tools
"""
import os
import sqlite3

# langchain_core.tools is a submodule that ships several tool-related utilities:
#   tool            -- decorator used here: wraps a plain function into a LangChain Tool object
#   BaseTool        -- base class all tools inherit from (used internally by @tool)
#   StructuredTool  -- explicit alternative to @tool when you need more control
#   InjectedToolArg -- marks arguments the system fills in, not the LLM
# We only import `tool` because the @tool decorator handles everything automatically.
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL, CLASSIFIER_TEMPERATURE,
    DB_PATH, MAX_TOKENS, MODEL_NAME, TEMPERATURE,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
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
    temperature=CLASSIFIER_TEMPERATURE,
    max_tokens=CLASSIFIER_MAX_TOKENS,
    reasoning_effort="low",
)


# ---------------------------------------------------------------------------
# TODO 1 of 4 -- implement query_doctors()
# ---------------------------------------------------------------------------
# query_doctors() fetches doctor information from the Apollo Health Clinic database.
#
# Decorate with @tool and give it a clear docstring -- the docstring becomes
# the tool's description in the JSON schema sent to the LLM. The LLM reads
# it to decide when and how to call the tool. Poor docstrings = wrong calls.
#
# Steps:
#
#   1. Decorate with @tool:
#        @tool
#        def query_doctors(department: str = "all") -> str:
#
#   2. Write a descriptive docstring explaining:
#        - what the function does (fetch doctor info from Apollo DB)
#        - what department accepts: a specialty name or "all"
#        - what the return value looks like
#
#   3. Open a SQLite connection (use check_same_thread=False):
#        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#
#   4. Query doctors:
#        - If department == "all": SELECT all doctors ordered by specialty, name
#        - Otherwise: filter with WHERE specialty LIKE ? (use f"%{department}%")
#        SELECT name, specialty, available_days, consultation_fee FROM doctors
#
#   5. Format each row as:
#        f"{name} | {specialty} | Available: {available_days} | Consultation: Rs.{fee}"
#
#   6. Close connection and return "\n".join(lines) or "No doctors found."
#
# WHY check_same_thread=False?
# SQLite raises ProgrammingError if a connection is used from a different thread
# than the one that created it. LangGraph runs tool calls in a thread-pool
# executor, so the tool may run on a different thread. This flag disables the check.
# ---------------------------------------------------------------------------

# TODO 1: implement query_doctors() below -- keep the @tool decorator and docstring
raise NotImplementedError("TODO 1: implement query_doctors() in clinicaliq/tools.py")


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- implement query_services()
# ---------------------------------------------------------------------------
# query_services() fetches diagnostic service and test information from the database.
#
# Steps:
#
#   1. Decorate with @tool:
#        @tool
#        def query_services(department: str = "all") -> str:
#
#   2. Write a descriptive docstring explaining:
#        - what the function does (fetch service info)
#        - what department accepts: a department name or "all"
#        - what the return value looks like
#
#   3. Open connection (check_same_thread=False):
#        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#
#   4. Query services:
#        - If department == "all": SELECT all services ordered by department, name
#        - Otherwise: filter with WHERE department LIKE ? (use f"%{department}%")
#        SELECT name, department, average_duration_mins, price FROM services
#
#   5. Format each row as:
#        f"{name} | {dept} | Duration: {duration} min | Price: Rs.{price}"
#
#   6. Close connection and return "\n".join(lines) or "No services found."
#
# ---------------------------------------------------------------------------

# TODO 2: implement query_services() below -- keep the @tool decorator and docstring
raise NotImplementedError("TODO 2: implement query_services() in clinicaliq/tools.py")


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- TOOLS list + llm_with_tools + _run_tool()
# ---------------------------------------------------------------------------
# Three steps:
#
#   Step A: build the TOOLS list
#        TOOLS = [
#            query_doctors,
#            query_services,
#        ]
#
#   Step B: create llm_with_tools
#     bind_tools() reads each function's type hints and docstring, converts them
#     to a JSON schema, and includes that schema in every request sent to the LLM.
#
#        llm_with_tools = llm.bind_tools(TOOLS)
#
#   Step C: implement _run_tool()
#
#        def _run_tool(tool_name: str, tool_args: dict) -> str:
#
#     Build a registry dict from TOOLS:
#        _registry = {t.name: t for t in TOOLS}
#     Return an error string if the tool name is unknown.
#     Otherwise call the tool's .invoke() method:
#        return _registry[tool_name].invoke(tool_args)
#     Wrap in try/except and return an error string on failure.
#
# WHY .invoke() and not a plain function call?
#   .invoke() validates arguments against the tool's schema before executing,
#   and it integrates with LangSmith — the tool call appears as a separate trace span.
# ---------------------------------------------------------------------------

# TODO 3: add TOOLS list, llm_with_tools, and _run_tool() below
raise NotImplementedError("TODO 3: add TOOLS, llm_with_tools, and _run_tool() in clinicaliq/tools.py")
