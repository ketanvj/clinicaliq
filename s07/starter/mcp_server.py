"""
ClinicalIQ -- Session 7: MCP Server (US-06 Part 1)
===================================================
STARTER FILE -- your task is to implement the two TODO sections below.

Goal
  Build a standalone MCP server that exposes ClinicalIQ's two database
  tools -- query_doctors and query_services -- over the MCP protocol.
  When finished, MCP Inspector should be able to discover both tools and
  call them without touching any agent code.

What is already done for you
  - FastMCP server created: mcp = FastMCP("clinicaliq-tools")
  - Both @mcp.tool() decorators and function signatures are in place
  - DB_PATH points to the same clinic_data.db used in Session 5
  - mcp.run() at the bottom starts the STDIO server

Your task
  Implement the SQL queries inside TODO 1 (query_doctors) and TODO 2
  (query_services). The logic is identical to s05/solution/clinicaliq/tools.py --
  open that file, find the two @tool functions, and adapt them here.
  The only change: replace @tool with @mcp.tool() (already done).

Run when done
  python s07/starter/mcp_server.py

Inspect with MCP Inspector
  npx @modelcontextprotocol/inspector python s07/starter/mcp_server.py
  Open http://localhost:5173 -- both tools should appear.
"""

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instantiation -- already done for you
# ---------------------------------------------------------------------------

mcp = FastMCP("clinicaliq-tools")

# ---------------------------------------------------------------------------
# Configuration -- already done for you
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH  = DATA_DIR / "clinic_data.db"

# ---------------------------------------------------------------------------
# TODO 1: Implement query_doctors
# Hint: copy the query_doctors() function from s05/solution/clinicaliq/tools.py.
#       The SQL queries and return format are identical.
#       The only difference: @tool becomes @mcp.tool() (already in place).
# ---------------------------------------------------------------------------

@mcp.tool()
def query_doctors(specialty: str = "all") -> str:
    """Fetch Apollo Health Clinic doctors and their consultation fees from the database.

    Args:
        specialty: Filter by medical specialty (e.g. "Cardiology", "General Medicine").
                   Use "all" to return every doctor.

    Returns formatted doctor information as a plain-text string.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    if specialty.lower() == "all":
        rows = conn.execute(
            "SELECT name, specialty, available_days, consultation_fee "
            "FROM doctors ORDER BY specialty"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, specialty, available_days, consultation_fee "
            "FROM doctors WHERE specialty LIKE ? ORDER BY name",
            (f"%{specialty}%",),
        ).fetchall()
    conn.close()
    if not rows:
        return f"No doctors found for specialty: '{specialty}'."
    parts = [
        f"{name} ({spec})\n  Available: {days} | Fee: Rs. {fee}"
        for name, spec, days, fee in rows
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# TODO 2: Implement query_services
# Hint: copy the query_services() function from s05/solution/clinicaliq/tools.py.
#       Same SQL, same return format, same @mcp.tool() decorator.
# ---------------------------------------------------------------------------

@mcp.tool()
def query_services(department: str = "all") -> str:
    """Fetch Apollo Health Clinic services and prices from the database.

    Args:
        department: Filter by department name (e.g. "Cardiology", "Diagnostics").
                    Use "all" to return every service.

    Returns formatted service and price information as a plain-text string.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    if department.lower() == "all":
        rows = conn.execute(
            "SELECT name, department, price FROM services ORDER BY department, price"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, department, price FROM services "
            "WHERE department LIKE ? ORDER BY price",
            (f"%{department}%",),
        ).fetchall()
    conn.close()
    if not rows:
        return f"No services found for department: '{department}'."
    lines = [f"{name} ({dept}): Rs. {price}" for name, dept, price in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point -- already done for you
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
