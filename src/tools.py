"""
tools.py
--------
A "tool" is just a Python function the LLM is allowed to call when it decides it needs to.
The @tool decorator from LangChain turns a normal function into something the LLM can see
and invoke. The docstring of each function is VERY important — the LLM reads it to decide
WHEN to use that tool, so always write clear, specific docstrings.
"""

import time
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from rag_pipeline import get_retriever

# ---------------------------------------------------------------------------
# TOOL 1: Web search — for questions about current events / anything outside
# the LLM's training data or your documents.
#
# We use Tavily here instead of scraping DuckDuckGo directly. DuckDuckGo's
# free endpoint frequently blocks automated/unauthenticated requests outright
# (you may see it fail 100% of the time depending on your network), whereas
# Tavily is a search API built specifically for AI agents, with a free tier
# (~1000 searches/month, no credit card). Get a free key at tavily.com.
# ---------------------------------------------------------------------------
_tavily = TavilySearch(max_results=4, topic="general")


@tool
def web_search(query: str) -> str:
    """Search the live web for current, real-time, or recent information
    (news, prices, weather, sports scores, anything that changes over time).
    Use this when the question is about something happening now or recently,
    or something you clearly don't know. Call this AT MOST ONCE per question —
    if it returns an error, tell the user search is unavailable rather than
    calling it again."""
    try:
        result = _tavily.invoke({"query": query})
        items = result.get("results", []) if isinstance(result, dict) else result
        if not items:
            return "No web results found for that query."
        return "\n\n".join(
            f"{item.get('title', '')}: {item.get('content', '')} "
            f"(source: {item.get('url', '')})"
            for item in items
        )
    except Exception as e:
        return (
            f"Web search is temporarily unavailable ({e}). "
            "Do not retry — tell the user the web search failed right now "
            "and answer from what you already know, if possible."
        )


# ---------------------------------------------------------------------------
# TOOL 2: Document retrieval (RAG) — for questions about the user's own documents.
# ---------------------------------------------------------------------------
_retriever = get_retriever(k=3)


@tool
def retrieve_documents(query: str) -> str:
    """Search the user's internal/private documents (e.g. company handbook,
    policies, notes) for information relevant to the query. Use this when the
    question sounds like it's about internal company info, policies, or any
    document the user might have uploaded."""
    results = _retriever.invoke(query)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(doc.page_content for doc in results)


# ---------------------------------------------------------------------------
# TOOL 3: Calculator — a tiny example of giving the LLM a "skill" it's bad at natively.
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '18% of 4500' style math
    problems written as a plain expression like '4500 * 0.18'. Use this for any
    math instead of trying to compute it yourself."""
    try:
        # NOTE: eval() is used here only for a simple local demo project.
        # In a production system, use a safe math parser (e.g. the `numexpr` library)
        # instead of eval() to avoid arbitrary code execution.
        allowed_chars = set("0123456789+-*/(). %")
        if not set(expression).issubset(allowed_chars):
            return "Error: expression contains characters that aren't allowed."
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# All tools the agent can choose from, gathered in one list
all_tools = [web_search, retrieve_documents, calculator]
