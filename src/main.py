"""
main.py
-------
Run this file to chat with your agent in the terminal:
    python src/main.py
"""

# IMPORTANT: load the .env file FIRST, before importing anything else from this
# project. agent_graph.py imports tools.py, which imports rag_pipeline.py, and
# rag_pipeline.py creates an OpenAIEmbeddings object the moment it's imported —
# so the API key has to already be in the environment before that import line runs.
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent_graph import build_graph


def main():
    print("=" * 60)
    print("🧠  Agentic RAG Research Assistant  (type 'exit' to quit)")
    print("=" * 60)

    app = build_graph()

    # thread_id identifies ONE conversation. Use a different thread_id to start
    # a fresh conversation with no memory of this one.
    config = {"configurable": {"thread_id": "demo-conversation-1"}}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not user_input:
            continue

        # Stream the graph's execution so we can see which tool it decides to use
        try:
            final_answer = None
            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values",
            ):
                last_message = event["messages"][-1]
                if getattr(last_message, "tool_calls", None):
                    for call in last_message.tool_calls:
                        print(f"   🔧 calling tool: {call['name']}({call['args']})")
                final_answer = last_message

            print(f"\nAssistant: {final_answer.content}")
        except Exception as e:
            print(f"\n⚠️  Something went wrong on that turn: {e}")
            print("You can try rephrasing your question, or just ask again.")


if __name__ == "__main__":
    main()
