# 🧠 Agentic RAG Research Assistant

A resume-ready AI Engineering project built with **LangChain**, **LangGraph**, and **FAISS**.

It's a chatbot that intelligently decides — for every question you ask — whether it should:
1. **Search your own documents** (RAG / Retrieval-Augmented Generation), or
2. **Search the live web** (via a tool call), or
3. **Do a calculation** (via a calculator tool), or
4. Just **answer directly** from its own knowledge.

It also **remembers the conversation** across turns (persistent memory), the same way ChatGPT
remembers what you said two messages ago.

This is exactly the kind of project AI Engineering job postings ask for: agentic workflows,
tool-calling, RAG, and graph-based orchestration.

---

## 🏗️ Architecture

```
                 ┌─────────────┐
                 │   You ask   │
                 │  a question │
                 └──────┬──────┘
                        ▼
                ┌───────────────┐
        ┌───────┤   AGENT NODE  │◄────────────┐
        │       │ (the LLM      │              │
        │       │  decides what │              │
        │       │  to do next)  │              │
        │       └───────┬───────┘              │
        │               │                      │
        │   Does it need a tool?               │
        │      /              \                │
        │    YES               NO              │
        │     │                 │              │
        │     ▼                 ▼              │
        │ ┌─────────┐      ┌─────────┐         │
        │ │TOOL NODE│      │   END   │         │
        │ │ runs:   │      │ (final  │         │
        │ │•retrieve│      │ answer) │         │
        │ │•web     │      └─────────┘         │
        │ │ search  │                          │
        │ │•calc    │                          │
        │ └────┬────┘                          │
        │      │  (tool result goes back        │
        └──────┴──  to the agent to reason again)
```

This loop (Agent → Tool → Agent → Tool → ... → END) is a **graph**, which is exactly what
LangGraph is built to model. A plain "chain" can't loop or make decisions like this — that's
why we need a graph instead of a simple LangChain chain.

---

## 📁 Project Structure

```
ai-research-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/
│       └── company_handbook.txt   ← swap this with your own PDFs/txt files
└── src/
    ├── rag_pipeline.py   ← loads docs, chunks them, embeds them, builds FAISS index
    ├── tools.py          ← defines the tools the agent can call
    ├── agent_graph.py     ← the LangGraph state machine (the "brain")
    └── main.py            ← CLI chat loop, entry point
```

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free API keys
cp .env.example .env
# Groq (chat model): get a free key at https://console.groq.com/keys
# Tavily (web search): get a free key at https://app.tavily.com
# open .env and paste both in

# 4. Run it (terminal version)
python src/main.py

# ...or run the web UI instead
streamlit run src/streamlit_app.py
```

This project runs **entirely for free**:
- The chat model uses **Groq**, which hosts open models (Llama 3.1) and has a free tier with no credit card required.
- Web search uses **Tavily**, a search API built for AI agents, with a free tier (~1000 searches/month, no credit card).
- The RAG embeddings use a **local HuggingFace model** (`all-MiniLM-L6-v2`) that downloads once (~90MB) and then runs on your own machine — no API key or cost for that step at all.

Then just chat in your terminal:

```
You: What is our company's refund policy?
[Assistant searches your documents → answers from the handbook]

You: What's the weather like in Bengaluru right now?
[Assistant realizes it doesn't know → calls the web search tool → answers]

You: What's 18% of 4500?
[Assistant calls the calculator tool → answers]

You: What did I just ask you about the weather?
[Assistant remembers, because the graph has memory]
```

### Using OpenAI or Claude instead of Groq
This project uses `ChatGroq` (free) by default. To use a paid provider instead, swap the
import in `src/agent_graph.py`, e.g. for Claude:
```python
# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama-3.3-70b-versatile")
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
```
and `pip install langchain-anthropic`, then set `ANTHROPIC_API_KEY` in `.env` instead. This is
a good thing to mention in an interview — it shows you understand LangChain is *model-agnostic*:
swapping providers is a one-line change because they all implement the same interface.

---

## 🎓 Key Concepts Demonstrated (great for interviews)

| Concept | Where it lives | Why it matters |
|---|---|---|
| RAG (Retrieval-Augmented Generation) | `rag_pipeline.py` | Lets the LLM answer from *your* private data instead of only its training data |
| Tool calling / function calling | `tools.py` | Lets the LLM take actions in the real world (search, calculate) |
| Agentic routing | `agent_graph.py` | The LLM *decides* what to do next instead of following a fixed script |
| Graph-based orchestration | `agent_graph.py` | Handles loops and branches that a simple linear chain can't |
| Persistent memory / checkpointing | `agent_graph.py`, `MemorySaver` | The assistant remembers earlier turns in the conversation |
| Vector embeddings + similarity search | `rag_pipeline.py` (FAISS) | The core technology behind semantic search |

---

## 🚀 Ideas to extend it (mention these in your interview even if you don't build them)
- Swap FAISS for a hosted vector DB (Pinecone, Chroma Cloud, Weaviate)
- Add a "human-in-the-loop" node so the agent asks for approval before big actions
- Add LangSmith for tracing/debugging the agent's reasoning steps
- Add a second specialized agent (e.g., a "coding agent") and route between them (multi-agent system)

---

## 🌐 Hosting the Streamlit app for free (Streamlit Community Cloud)

This gives you a live public URL to put on your resume/LinkedIn instead of just a local demo.

1. **Push this project to a public GitHub repo.**
   ```bash
   git init
   git add .
   git commit -m "Agentic RAG Research Assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   The included `.gitignore` already keeps your `.env` file and local FAISS index out of the repo — don't remove those lines.

2. **Go to https://share.streamlit.io** and sign in with GitHub.

3. Click **"Create app"** → **"Deploy a public app from GitHub"**, then select:
   - Repository: your repo
   - Branch: `main`
   - Main file path: `src/streamlit_app.py`

4. Before clicking deploy, open **"Advanced settings"** and paste your keys into the **Secrets** box in TOML format:
   ```toml
   GROQ_API_KEY = "gsk-your-key-here"
   TAVILY_API_KEY = "tvly-your-key-here"
   ```
   (This is the cloud equivalent of your local `.env` file — `streamlit_app.py` is already written to pick these up automatically.)

5. Click **Deploy**. First deploy takes a few minutes (installing dependencies + downloading the small embedding model). You'll get a URL like `https://your-app-name.streamlit.app` that you can share with anyone.

**Note:** the free tier sleeps after inactivity and wakes up on the next visit (takes ~30–60 seconds) — completely normal and fine for a resume demo link.

