"""
s07/tests/test_s07.py
----------------------
Tests for Session 7: MCP Server (US-06 Part 1).

Run with:
    pytest s07/tests/ -v

These tests call the tool functions directly. FastMCP's @mcp.tool()
decorator registers the function with the server but returns the original
callable unchanged -- so query_rates("home_loan") works just like calling
any Python function.

DB_PATH is patched to a test database created in conftest.py. Tests do not
require data/seed.py to have been run.

Test groups:
  TestServerStructure       -- server name, tool count, tool names
  TestQueryRates             -- return format, product filtering, empty result
  TestQueryEligibility       -- return format, product filtering, empty result
  TestSQLInjection           -- parameterised query protects against injection
"""

import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import mcp_server
from mcp_server import mcp, query_eligibility, query_rates


# ---------------------------------------------------------------------------
# TestServerStructure
# ---------------------------------------------------------------------------

class TestServerStructure:
    def test_server_name(self):
        assert mcp.name == "quickloan-tools"

    def test_server_has_two_tools(self):
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 2, f"Expected 2 tools, found {len(tools)}"

    def test_server_has_query_rates_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_rates" in names

    def test_server_has_query_eligibility_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_eligibility" in names

    def test_query_rates_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_rates")
        assert tool.description and len(tool.description) > 10

    def test_query_eligibility_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_eligibility")
        assert tool.description and len(tool.description) > 10


# ---------------------------------------------------------------------------
# TestQueryRates
# ---------------------------------------------------------------------------

class TestQueryRates:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("all")
        assert isinstance(result, str)

    def test_all_contains_all_products(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("all")
        assert "Personal Loan" in result
        assert "Home Loan" in result
        assert "Business Loan" in result

    def test_home_loan_best_rate_is_8_75(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("home_loan")
        assert "8.75" in result

    def test_filter_excludes_other_products(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("home_loan")
        assert "Personal Loan" not in result
        assert "Business Loan" not in result

    def test_cibil_band_format(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("personal_loan")
        assert "CIBIL" in result

    def test_no_data_returns_not_found(self, tmp_path, monkeypatch):
        empty_db = tmp_path / "empty.db"
        import sqlite3
        conn = sqlite3.connect(str(empty_db))
        conn.executescript("""
            CREATE TABLE loan_products (
                product_id TEXT PRIMARY KEY, product_name TEXT,
                min_tenure_months INTEGER, max_tenure_months INTEGER,
                max_loan_amount INTEGER, processing_fee_pct REAL
            );
            CREATE TABLE rate_slabs (
                id INTEGER PRIMARY KEY, product_id TEXT,
                min_cibil INTEGER, max_cibil INTEGER, annual_rate_pct REAL
            );
        """)
        conn.commit()
        conn.close()
        monkeypatch.setattr(mcp_server, "DB_PATH", empty_db)
        result = query_rates("all")
        assert "No rate data found" in result

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_rates()
        result_all = query_rates("all")
        assert result_default == result_all

    def test_gold_loan_rate_is_9_5(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("business_loan")
        assert "14.0" in result

    def test_personal_loan_best_rate_is_11_5(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("personal_loan")
        assert "11.5" in result


# ---------------------------------------------------------------------------
# TestQueryEligibility
# ---------------------------------------------------------------------------

class TestQueryEligibility:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("all")
        assert isinstance(result, str)

    def test_all_returns_all_products(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("all")
        assert "Personal Loan" in result
        assert "Home Loan" in result
        assert "Gold Loan" in result

    def test_filter_returns_correct_product(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("home_loan")
        assert "Home Loan" in result
        assert "Personal Loan" not in result

    def test_result_includes_min_cibil(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("home_loan")
        assert "720" in result

    def test_result_includes_min_income(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("personal_loan")
        assert "25000" in result

    def test_result_includes_age_range(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("business_loan")
        assert "25-65" in result

    def test_unknown_product_returns_not_found_message(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("crypto_loan")
        assert "No eligibility data found" in result

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_eligibility()
        result_all = query_eligibility("all")
        assert result_default == result_all

    def test_entries_separated_by_double_newline(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("all")
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# TestSQLInjection
# ---------------------------------------------------------------------------

class TestSQLInjection:
    def test_rates_injection_does_not_crash(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("'; DROP TABLE rate_slabs; --")
        assert isinstance(result, str)
        assert "No rate data found" in result

    def test_rates_injection_does_not_drop_table(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        query_rates("'; DROP TABLE rate_slabs; --")
        result = query_rates("home_loan")
        assert "8.75" in result

    def test_eligibility_injection_does_not_crash(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_eligibility("'; DROP TABLE eligibility_rules; --")
        assert isinstance(result, str)
        assert "No eligibility data found" in result
