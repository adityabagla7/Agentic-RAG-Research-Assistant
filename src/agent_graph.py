"""
agent_graph.py
--------------
This is the "brain" of the project, built with LangGraph.

Why LangGraph and not just LangChain?
  A plain LangChain "chain" runs in a straight line: Step 1 -> Step 2 -> Step 3 -> done.
  But a real agent needs to LOOP: ask the LLM what to do -> maybe call a tool -> show the
  LLM the tool's result -> ask again -> maybe call another tool -> ... -> finally answer.
  That loop, with branches ("if a tool is needed, go here, otherwise go there"), is a GRAPH,
  not a straight line. LangGraph lets us define exactly that graph: nodes (steps) and edges
  (the rules for which node runs next).
"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict

# Load .env BEFORE importing tools (which imports rag_pipeline, which needs
# OPENAI_API_KEY to already be set the moment it's imported).
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools import all_tools


# ---------------------------------------------------------------------------
# 1. STATE — the "shared notebook" that gets passed between every node in the graph.
#    Here it's just a running list of chat messages. `add_messages` is a special
#    reducer that means "append new messages to the list" instead of overwriting it.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 2. THE LLM — the model doing the reasoning. `.bind_tools()` tells it which
#    tools exist and their descriptions, so it can decide to call one.
#    Groq hosts open models and gives a generous free tier — no credit card
#    required. Get a key at https://console.groq.com/keys
#
#    NOTE: we use llama-3.1-8b-instant rather than the 70B model here. Larger
#    Llama models on Groq can occasionally emit malformed tool-call syntax
#    when multiple tools are bound at once (you may see a "tool call
#    validation failed" error) — the 8b-instant model is smaller but much
#    more consistent at correctly formatted tool calls, which matters more
#    than raw intelligence for an agent like this one.
# ---------------------------------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
# parallel_tool_calls=False: ask the model to request one tool at a time
# instead of several at once, which is the main thing that triggers malformed
# tool-call output on Groq's hosted Llama models.
llm_with_tools = llm.bind_tools(all_tools, parallel_tool_calls=False)

SYSTEM_PROMPT = """You are a helpful AI research assistant.
You have access to tools: web_search, retrieve_documents, and calculator.
- Use retrieve_documents for questions about the user's internal documents/policies.
- Use web_search for current events or anything you don't know.
- Use calculator for any math.
- If you don't need a tool, just answer directly.
- Call each tool AT MOST ONCE per question. If a tool returns an error or says
  it's unavailable, do NOT call it again — tell the user it failed and answer
  from what you already know, if you can.
Always answer clearly and concisely."""


# ---------------------------------------------------------------------------
# 3. NODES — the individual steps in the graph.
# ---------------------------------------------------------------------------
def agent_node(state: AgentState):
    """The 'thinking' step: the LLM looks at the conversation so far and decides
    whether to respond directly or call a tool."""
    from langchain_core.messages import SystemMessage, AIMessage

    messages = state["messages"]
    # Prepend the system prompt only once, at the start of the conversation
    if not any(getattr(m, "type", None) == "system" for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        # Occasionally a model emits a malformed tool call and the API rejects
        # it (a "tool call validation failed" / 400 error). Instead of crashing
        # the whole program, retry once without tools so the user still gets
        # an answer.
        print(f"   ⚠️  Tool call failed ({e}), retrying without tools...")
        response = llm.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(all_tools)


# ---------------------------------------------------------------------------
# 4. BUILD THE GRAPH — wire the nodes together with edges.
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    # After the agent thinks: if it wants to call a tool, go to "tools",
    # otherwise the conversation is done (END). `tools_condition` is a
    # ready-made LangGraph helper that checks this automatically.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    # After a tool runs, go back to the agent so it can read the result and continue
    graph.add_edge("tools", "agent")

    # MemorySaver = a checkpointer that stores conversation state, so the agent
    # remembers earlier turns as long as you keep passing the same thread_id.
    memory = MemorySaver()

    return graph.compile(checkpointer=memory)
