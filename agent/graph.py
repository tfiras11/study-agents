"""The assembly: wire the agents into a LangGraph machine.

This file contains no intelligence -- it is pure wiring. Read it as the
answer to: "WHO runs after WHOM, and WHERE does memory come from?"

    START ──▶ supervisor ──(conditional edge on state["route"])──┐
                        │              ┌───────────┬───────────┤
                        ▼              ▼           ▼           ▼
                      tutor          quiz      planner       chat
                        └──────────────┴───────────┴───────────┘
                                       ▼
                                      END  ──▶ checkpointer saves state

One deterministic shortcut before the supervisor: if a quiz question is on
the table, the student's message IS an answer -- we skip the LLM call.

TWO ways this graph gets compiled, for two different runtimes:

- `build_graph()` -- used by app.py (local Streamlit). Compiles WITH a
  local SqliteSaver so a conversation survives an app restart on your
  machine.
- `graph`         -- the module-level export LangSmith Deployment serves
  (see langgraph.json). Compiled WITHOUT a checkpointer: the platform
  injects its own managed, persistent one at deploy time. Passing our own
  SqliteSaver here would conflict with that and doesn't even make sense on
  a hosted server with an ephemeral/scaled-out filesystem.
"""
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

from config import CHECKPOINT_PATH, CHAT_MODEL, NVIDIA_API_KEY
from agent.state import AgentState
from agent.router import supervisor_node
from agent.tutor import tutor_node
from agent.quiz import quiz_node
from agent.planner import planner_node

CHAT_PROMPT = """Tu es l'assistant d'une plateforme d'étude pour un élève
tunisien de 1ère année secondaire. Sois bref, chaleureux, et encourage-le
doucement à étudier. Si sa question concerne une leçon, dis-lui que le
tuteur peut l'aider."""


def chat_node(state, config):
    """Small talk: no RAG, no structured output, just a friendly reply."""
    llm = ChatNVIDIA(model=CHAT_MODEL, temperature=0.6, api_key=NVIDIA_API_KEY)
    reply = llm.invoke(
        [SystemMessage(content=CHAT_PROMPT)] + state["messages"]
    )
    return {"messages": [reply], "sources": []}


def supervisor_with_override(state, config):
    """Deterministic shortcut: when a quiz question awaits, the student's
    message IS an answer -> route to quiz WITHOUT any LLM call."""
    if state.get("awaiting_answer"):
        return {"route": "quiz"}
    return supervisor_node(state, config)


def route_after_supervisor(state) -> str:
    """The conditional edge: reads state, returns the NEXT node's name."""
    return state["route"]


def _make_builder() -> StateGraph:
    """Wire the nodes and edges. Shared by both compile paths below so the
    graph topology can never drift between local and deployed runs."""
    graph = StateGraph(AgentState)  # the graph is typed by our whiteboard

    graph.add_node("supervisor", supervisor_with_override)
    graph.add_node("tutor", tutor_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("planner", planner_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "supervisor")  # every turn starts at the router
    graph.add_conditional_edges(
        "supervisor",                    # source node
        route_after_supervisor,          # function that reads state
        {                                # return value -> destination node
            "tutor": "tutor",
            "quiz": "quiz",
            "planner": "planner",
            "chat": "chat",
        },
    )
    for name in ("tutor", "quiz", "planner", "chat"):
        graph.add_edge(name, END)        # one agent runs, then the turn ends

    return graph


def build_graph():
    """Local/Streamlit path: compile WITH a local SQLite checkpointer.

    Memory: every turn's full state is saved to this SQLite file, keyed
    by thread_id. check_same_thread=False lets Streamlit's threads share
    the connection.
    """
    conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    return _make_builder().compile(checkpointer=SqliteSaver(conn))


# --- Deploy path: what LangSmith Deployment actually serves -------------
# langgraph.json points at this module-level name ("./agent/graph.py:graph").
# No checkpointer is passed -- LangSmith Deployment provides its own
# managed, persistent one automatically for every deployed graph.
graph = _make_builder().compile()


def run_agent(graph, user_text: str, subject: str, thread_id: str) -> dict:
    """One student turn: send a message, get the reply + debug info back.

    thread_id is THE memory key: same id = same continuing conversation;
    a new id starts a fresh one (each subject gets its own).

    Works against either compile path above (local build_graph() or the
    deployed `graph`) -- it just needs an object with .invoke().
    """
    config = {"configurable": {"thread_id": thread_id, "subject": subject}}
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config)
    return {
        "reply": result["messages"][-1].content,
        "route": result.get("route", "chat"),
        "sources": result.get("sources", []),
        "asked": result.get("asked", 0),
        "score": result.get("score", 0),
    }