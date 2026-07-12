"""
streamlit_app.py
-----------------
A web UI for the agent, built with Streamlit. This is what you'd screenshot or
screen-record for a resume/LinkedIn demo instead of a plain terminal.

Run with:
    streamlit run src/streamlit_app.py
"""

import os
import uuid
from dotenv import load_dotenv
load_dotenv()  # for local development (.env file)

import streamlit as st

# On Streamlit Community Cloud there's no .env file — secrets are provided via
# st.secrets instead (set in the app's "Secrets" settings after deploying).
# This copies them into the environment so agent_graph.py / tools.py, which
# just read os.environ, work identically whether run locally or deployed.
for _key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    if not os.environ.get(_key):
        try:
            if _key in st.secrets:
                os.environ[_key] = st.secrets[_key]
        except Exception:
            pass  # no secrets.toml present (e.g. local run using only .env) — fine

from langchain_core.messages import HumanMessage, AIMessage
from agent_graph import build_graph

st.set_page_config(page_title="Agentic RAG Research Assistant", page_icon="🧠", layout="centered")

st.title("🧠 Agentic RAG Research Assistant")
st.caption(
    "An AI agent built with LangChain + LangGraph. It decides on its own whether to "
    "search your documents, search the web, do a calculation, or answer directly."
)

# ---------------------------------------------------------------------------
# SESSION STATE — Streamlit re-runs this whole script on every interaction, so
# anything that needs to persist across turns (the graph, the thread_id, the
# chat history for display) has to live in st.session_state.
# ---------------------------------------------------------------------------
if "graph_app" not in st.session_state:
    st.session_state.graph_app = build_graph()

if "thread_id" not in st.session_state:
    # A unique id per browser session = a unique memory thread for LangGraph
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    # Just for rendering the chat bubbles — separate from the graph's own
    # internal message state, which LangGraph tracks itself via the checkpointer
    st.session_state.display_messages = []

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Tools available to the agent:**
        - 📄 `retrieve_documents` — searches `data/sample_docs` (RAG / FAISS)
        - 🌐 `web_search` — live web search (Tavily)
        - 🔢 `calculator` — arithmetic

        **Try asking:**
        - *"What's the refund policy?"* → uses RAG
        - *"What's the latest news on X?"* → uses web search
        - *"What's 18% of 4500?"* → uses calculator
        """
    )
    if st.button("🔄 Start new conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# RENDER CHAT HISTORY
# ---------------------------------------------------------------------------
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                st.caption(f"🔧 used tool: `{tc}`")

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tool_calls_used = []
            final_answer = None
            try:
                for event in st.session_state.graph_app.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="values",
                ):
                    last_message = event["messages"][-1]
                    if getattr(last_message, "tool_calls", None):
                        for call in last_message.tool_calls:
                            tool_calls_used.append(f"{call['name']}({call['args']})")
                    final_answer = last_message

                answer_text = final_answer.content
            except Exception as e:
                answer_text = f"⚠️ Something went wrong: {e}"

        st.markdown(answer_text)
        for tc in tool_calls_used:
            st.caption(f"🔧 used tool: `{tc}`")

    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer_text, "tool_calls": tool_calls_used}
    )
