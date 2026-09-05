"""
s04/tests/test_s04.py
---------------------
Unit tests for Session 4: ChromaDB RAG.

Run from the quickloan/ directory:
    pytest s04/tests/ -v

All tests run without a live Groq API key, ChromaDB, or HuggingFace model.
The vectorstore and LLMs are mocked throughout.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "quickloan" or _k.startswith("quickloan."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from quickloan.config import DECLINE_RESPONSE, ESCALATE_RESPONSE, RETRIEVAL_K, SYSTEM_PROMPT  # noqa: E402
from quickloan.state import QuickLoanState  # noqa: E402
import quickloan.nodes as _nodes  # noqa: E402
from quickloan.nodes import classify, decline, escalate, respond, retrieve_docs, route_query  # noqa: E402
from quickloan.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_doc(page_content: str, source: str = "home_loan_guide.md") -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": source}
    return doc


def _mock_vectorstore_with(docs: list) -> MagicMock:
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = docs
    return mock_vs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def mock_llm_simple():
    with patch.object(_nodes, "llm") as mock_main, \
         patch.object(_nodes, "classifier_llm") as mock_clf:
        mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
        mock_main.invoke.return_value = MagicMock(
            content="For a home loan, you need salary slips and a PAN card. QuickLoan | FastFinance India"
        )
        yield mock_main, mock_clf


@pytest.fixture
def mock_vectorstore():
    doc = _make_mock_doc(
        "For a home loan at FastFinance India, you need: last 6 months' salary slips, "
        "PAN card, Aadhaar card, Form 16, and 6 months' bank statements.",
        source="home_loan_guide.md",
    )
    mock_vs = _mock_vectorstore_with([doc])
    with patch.object(_nodes, "vectorstore", mock_vs), \
         patch.object(_nodes, "_init_vectorstore"):
        yield mock_vs


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestQuickLoanState:
    def test_state_has_customer_message(self):
        assert "customer_message" in QuickLoanState.__annotations__

    def test_state_has_response(self):
        assert "response" in QuickLoanState.__annotations__

    def test_state_has_history(self):
        assert "history" in QuickLoanState.__annotations__

    def test_state_has_query_type(self):
        assert "query_type" in QuickLoanState.__annotations__

    def test_state_has_retrieved_docs(self):
        assert "retrieved_docs" in QuickLoanState.__annotations__, (
            "QuickLoanState must have a 'retrieved_docs' field (added in Session 4). "
            "Add it after 'query_type' with type hint list[str]."
        )

    def test_retrieved_docs_is_list_type(self):
        annotation = QuickLoanState.__annotations__["retrieved_docs"]
        origin = getattr(annotation, "__origin__", annotation)
        assert origin is list

    def test_state_instantiable_with_all_fields(self):
        state: QuickLoanState = {
            "customer_message": "What documents do I need for a home loan?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
        }
        assert state["retrieved_docs"] == []

    def test_state_retrieved_docs_accepts_strings(self):
        state: QuickLoanState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   ["[home_loan_guide.md]\nFor a home loan, you need salary slips."],
        }
        assert len(state["retrieved_docs"]) == 1


# ---------------------------------------------------------------------------
# retrieve_docs() node tests
# ---------------------------------------------------------------------------

class TestRetrieveDocsNode:
    def _state(self, question: str = "What documents do I need for a home loan?") -> QuickLoanState:
        return {
            "customer_message": question,
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
        }

    def test_retrieve_docs_returns_dict(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result, dict)

    def test_retrieve_docs_returns_retrieved_docs_key(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert "retrieved_docs" in result

    def test_retrieve_docs_returns_list(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result["retrieved_docs"], list)

    def test_retrieve_docs_calls_similarity_search(self, mock_vectorstore):
        retrieve_docs(self._state("What documents do I need?"))
        mock_vectorstore.similarity_search.assert_called_once()

    def test_retrieve_docs_passes_question_to_search(self, mock_vectorstore):
        question = "What documents do I need for a home loan?"
        retrieve_docs(self._state(question))
        call_args = mock_vectorstore.similarity_search.call_args
        assert call_args[0][0] == question or call_args[1].get("query") == question

    def test_retrieve_docs_passes_k_parameter(self, mock_vectorstore):
        retrieve_docs(self._state())
        call_args = mock_vectorstore.similarity_search.call_args
        k_value   = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("k")
        assert k_value == RETRIEVAL_K

    def test_retrieve_docs_formats_source_in_output(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert len(result["retrieved_docs"]) > 0
        first = result["retrieved_docs"][0]
        assert "home_loan_guide.md" in first

    def test_retrieve_docs_includes_page_content(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        combined = " ".join(result["retrieved_docs"])
        assert "salary" in combined or "home loan" in combined.lower()

    def test_retrieve_docs_returns_empty_when_vectorstore_is_none(self):
        with patch.object(_nodes, "vectorstore", None), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}

    def test_retrieve_docs_returns_empty_on_exception(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = Exception("ChromaDB connection error")
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}

    def test_retrieve_docs_does_not_call_llm(self, mock_vectorstore):
        with patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "classifier_llm") as mock_clf:
            retrieve_docs(self._state())
            mock_llm.invoke.assert_not_called()
            mock_clf.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# respond() node tests -- context injection
# ---------------------------------------------------------------------------

class TestRespondWithContext:
    def _state_with_docs(self, docs: list[str]) -> QuickLoanState:
        return {
            "customer_message": "What documents do I need for a home loan?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   docs,
        }

    def test_respond_includes_retrieved_docs_in_system_message(self):
        chunk = "[home_loan_guide.md]\nFor a home loan: salary slips, PAN card, Aadhaar."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="You need salary slips and PAN.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "salary" in system_text or "home_loan_guide.md" in system_text

    def test_respond_without_docs_escalates_directly(self):
        state = self._state_with_docs([])
        with patch.object(_nodes, "llm") as mock_llm:
            result = respond(state)
        mock_llm.invoke.assert_not_called()
        assert result["response"] == ESCALATE_RESPONSE

    def test_respond_context_contains_policy_keyword(self):
        chunk = "[home_loan_guide.md]\nSalary slips and PAN card required for home loan."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Salary slips required.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "salary" in system_text.lower() or "PAN" in system_text

    def test_respond_with_multiple_docs(self):
        chunks = [
            "[home_loan_guide.md]\nSalary slips required for home loan application.",
            "[fastfinance_policy.md]\nProcessing fee of 1% applies on all loans.",
        ]
        state = self._state_with_docs(chunks)
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Salary slips needed; 1% fee applies.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "salary" in system_text.lower() or "Salary" in system_text
        assert "Processing" in system_text or "processing" in system_text

    def test_respond_updates_history_with_docs(self):
        chunk = "[faq.md]\nYou can apply for a loan online at fastfinance.in."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Apply online at fastfinance.in.")
            result = respond(state)
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# route_query() tests
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def _state(self, query_type: str) -> dict:
        return {"customer_message": "test", "response": "",
                "history": [], "query_type": query_type, "retrieved_docs": []}

    def test_simple_routes_to_retrieve_docs(self):
        assert route_query(self._state("SIMPLE")) == "retrieve_docs"

    def test_complex_routes_to_retrieve_docs(self):
        # Under Option B, COMPLEX is not a valid type — coerced to IN_SCOPE → retrieve_docs.
        assert route_query(self._state("COMPLEX")) == "retrieve_docs"

    def test_out_of_scope_routes_to_decline(self):
        assert route_query(self._state("OUT_OF_SCOPE")) == "decline"

    def test_default_routes_to_retrieve_docs(self):
        state = {"customer_message": "test", "response": "", "history": [], "retrieved_docs": []}
        assert route_query(state) == "retrieve_docs"


# ---------------------------------------------------------------------------
# Graph routing tests
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def test_simple_path_calls_vectorstore(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-simple-rag"}}
        graph.invoke(
            {"customer_message": "What documents do I need for a home loan?", "response": ""},
            config=config,
        )
        mock_vectorstore.similarity_search.assert_called_once()

    def test_simple_path_sets_retrieved_docs_in_result(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-simple-docs"}}
        result = graph.invoke(
            {"customer_message": "What documents do I need for a home loan?", "response": ""},
            config=config,
        )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) > 0

    def test_no_docs_path_escalates(self, memory_checkpointer):
        # Option B: IN_SCOPE → retrieve_docs → no docs returned → respond() escalates directly.
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "llm"):
            mock_clf.invoke.return_value = MagicMock(content="IN_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-no-docs-escalate"}}
            result = graph.invoke(
                {"customer_message": "Which plan suits heavy users?", "response": ""},
                config=config,
            )
        assert result["response"] == ESCALATE_RESPONSE

    def test_out_of_scope_path_skips_vectorstore(self, memory_checkpointer):
        mock_vs = MagicMock()
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "llm"):
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-oos-no-rag"}}
            graph.invoke(
                {"customer_message": "Tell me a joke.", "response": ""},
                config=config,
            )
        mock_vs.similarity_search.assert_not_called()

    def test_all_paths_produce_non_empty_response(self, mock_vectorstore, memory_checkpointer):
        for query_type, question in [
            ("SIMPLE",       "What documents do I need for a home loan?"),
            ("COMPLEX",      "Which loan is best for me?"),
            ("OUT_OF_SCOPE", "Tell me a joke."),
        ]:
            with patch.object(_nodes, "classifier_llm") as mock_clf, \
                 patch.object(_nodes, "llm") as mock_llm:
                mock_clf.invoke.return_value = MagicMock(content=query_type)
                mock_llm.invoke.return_value = MagicMock(content="Some answer.")
                graph  = build_graph(checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": f"all-paths-{query_type}"}}
                result = graph.invoke(
                    {"customer_message": question, "response": ""},
                    config=config,
                )
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemoryWithRAG:
    def test_history_accumulates_across_simple_turns(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "mem-rag-simple"
        config    = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {"customer_message": "What documents do I need for a home loan?", "response": ""},
            config=config,
        )
        result = graph.invoke(
            {"customer_message": "What is the personal loan interest rate?", "response": ""},
            config=config,
        )
        assert len(result["history"]) == 4

    def test_different_threads_isolated(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph = build_graph(checkpointer=memory_checkpointer)

        graph.invoke(
            {"customer_message": "What is the home loan rate?", "response": ""},
            config={"configurable": {"thread_id": "rag-thread-X"}},
        )
        result = graph.invoke(
            {"customer_message": "What documents do I need for a personal loan?", "response": ""},
            config={"configurable": {"thread_id": "rag-thread-Y"}},
        )
        assert len(result["history"]) == 2
