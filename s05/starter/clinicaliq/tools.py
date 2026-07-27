"""
clinicaliq/tools.py
-------------------
LLM clients and database tool functions for ClinicalIQ.

Session 5: adds query_doctors() and query_services() so the LLM can
look up live data instead of relying on hardcoded fees.
"""
import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import DB_PATH, MODEL_NAME, TEMPERATURE, MAX_TOKENS

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
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=10,
)


# ---------------------------------------------------------------------------
# TODO 1 of 4 -- Implement query_doctors()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. If specialty.lower() == "all":
#        rows = conn.execute(
#            "SELECT name, specialty, available_days, consultation_fee "
#            "FROM doctors ORDER BY specialty"
#        ).fetchall()
#      Otherwise use a parameterised query (CRITICAL -- prevents SQL injection):
#        rows = conn.execute(
#            "SELECT name, specialty, available_days, consultation_fee "
#            "FROM doctors WHERE specialty LIKE ? ORDER BY name",
#            (f"%{specialty}%",),
#        ).fetchall()
#   3. conn.close()
#   4. If not rows: return f"No doctors found for specialty: '{specialty}'."
#   5. Build parts = [] and for each row (name, spec, days, fee) append:
#        f"{name} ({spec})\n  Available: {days} | Fee: Rs. {fee}"
#      Return "\n\n".join(parts)
# ---------------------------------------------------------------------------
@tool
def query_doctors(specialty: str = "all") -> str:
    """Fetch Apollo Health Clinic doctors and their consultation fees from the database.

    Args:
        specialty: Filter by medical specialty (e.g. "Cardiology", "General Medicine").
                   Use "all" to return every doctor.

    Returns formatted doctor information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- Implement query_services()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. If department.lower() == "all":
#        rows = conn.execute(
#            "SELECT name, department, price FROM services ORDER BY department, price"
#        ).fetchall()
#      Otherwise use a parameterised query (CRITICAL -- prevents SQL injection):
#        rows = conn.execute(
#            "SELECT name, department, price FROM services "
#            "WHERE department LIKE ? ORDER BY price",
#            (f"%{department}%",),
#        ).fetchall()
#   3. conn.close()
#   4. If not rows: return f"No services found for department: '{department}'."
#   5. Build lines = [f"{name} ({dept}): Rs. {price}" for name, dept, price in rows]
#      Return "\n".join(lines)
# ---------------------------------------------------------------------------
@tool
def query_services(department: str = "all") -> str:
    """Fetch Apollo Health Clinic services and prices from the database.

    Args:
        department: Filter by department name (e.g. "Cardiology", "Diagnostics").
                    Use "all" to return every service.

    Returns formatted service and price information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- Bind tools to the LLM
# ---------------------------------------------------------------------------
# Create llm_with_tools by binding both tools to llm:
#   llm_with_tools = llm.bind_tools([query_doctors, query_services])
#
# This tells the LLM what tools are available so it can decide when to call them.
# llm_with_tools is used for the FIRST call in respond(). The second call
# (after tools have run) uses plain llm.
# ---------------------------------------------------------------------------
# TODO: add llm_with_tools = llm.bind_tools([query_doctors, query_services])


def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call by name. Provided -- no changes needed."""
    _registry = {
        "query_doctors":  query_doctors,
        "query_services": query_services,
    }
    if tool_name not in _registry:
        return f"Unknown tool: {tool_name}"
    try:
        return _registry[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
