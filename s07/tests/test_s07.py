"""
s07/tests/test_s07.py
----------------------
Tests for Session 7: MCP Server (US-06 Part 1).

Run with:
    pytest s07/tests/ -v

These tests call the tool functions directly. FastMCP's @mcp.tool()
decorator registers the function with the server but returns the original
callable unchanged -- so query_doctors("Cardiology") works just like calling
any Python function.

DB_PATH is patched to a test database created in conftest.py. Tests do not
require data/seed.py to have been run.

Test groups:
  TestServerStructure  -- server name, tool count, tool names
  TestQueryDoctors     -- return format, specialty filtering, empty result
  TestQueryServices    -- return format, department filtering, empty result
  TestSQLInjection     -- parameterised query protects against injection
"""

import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import mcp_server
from mcp_server import mcp, query_doctors, query_services


# ---------------------------------------------------------------------------
# TestServerStructure
# ---------------------------------------------------------------------------

class TestServerStructure:
    def test_server_name(self):
        assert mcp.name == "clinicaliq-tools"

    def test_server_has_two_tools(self):
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 2, f"Expected 2 tools, found {len(tools)}"

    def test_server_has_query_doctors_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_doctors" in names

    def test_server_has_query_services_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_services" in names

    def test_query_doctors_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_doctors")
        assert tool.description and len(tool.description) > 10

    def test_query_services_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_services")
        assert tool.description and len(tool.description) > 10


# ---------------------------------------------------------------------------
# TestQueryDoctors
# ---------------------------------------------------------------------------

class TestQueryDoctors:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("all")
        assert isinstance(result, str)

    def test_all_returns_multiple_doctors(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("all")
        assert "Dr. Meera Nair" in result
        assert "Dr. Rajesh Kumar" in result

    def test_specialty_filter_cardiology_returns_cardiology_only(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("Cardiology")
        assert "Rajesh Kumar" in result
        assert "Meera Nair" not in result

    def test_unknown_specialty_returns_not_found_message(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("Neurology")
        assert "No doctors found" in result
        assert "Neurology" in result

    def test_result_includes_fee(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("Cardiology")
        assert "Fee: Rs. 800" in result

    def test_result_includes_available_days(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("Cardiology")
        assert "Available" in result

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_doctors()
        result_all = query_doctors("all")
        assert result_default == result_all

    def test_case_insensitive_partial_match(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("cardiology")
        assert "Rajesh Kumar" in result or "No doctors found" in result

    def test_doctors_separated_by_double_newline(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("all")
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# TestQueryServices
# ---------------------------------------------------------------------------

class TestQueryServices:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("all")
        assert isinstance(result, str)

    def test_all_returns_multiple_services(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("all")
        assert "ECG" in result
        assert "Chest X-Ray" in result

    def test_department_filter_cardiology_only(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("Cardiology")
        assert "ECG" in result
        assert "Echocardiogram" in result
        assert "Chest X-Ray" not in result

    def test_unknown_department_returns_not_found_message(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("Neurology")
        assert "No services found" in result

    def test_result_includes_price(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("Cardiology")
        assert "350" in result

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_services()
        result_all = query_services("all")
        assert result_default == result_all

    def test_diagnostics_filter_returns_correct_services(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("Diagnostics")
        assert "Basic Blood Panel" in result
        assert "Urine Routine" in result


# ---------------------------------------------------------------------------
# TestSQLInjection
# ---------------------------------------------------------------------------

class TestSQLInjection:
    def test_doctors_injection_does_not_crash(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_doctors("'; DROP TABLE doctors; --")
        assert isinstance(result, str)
        assert "No doctors found" in result

    def test_doctors_injection_does_not_drop_table(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        query_doctors("'; DROP TABLE doctors; --")
        result = query_doctors("Cardiology")
        assert "Rajesh Kumar" in result

    def test_services_injection_does_not_crash(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_services("'; DROP TABLE services; --")
        assert isinstance(result, str)
        assert "No services found" in result
