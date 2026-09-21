"""The student-facing chat UI: one page, one chat, one subject at a time.

Run it like:
    streamlit run app.py

The whole intelligence lives in agent/graph.py -- this file is only a
shell around run_agent(): a subject picker, the chat transcript, and the
debug extras (sources, quiz score) shown where the student can use them.

Memory: the LangGraph checkpointer saves every turn to SQLite keyed by
thread_id, so a conversation survives app restarts. One thread per
subject; "Nouvelle conversation" starts a fresh thread.
"""
import uuid

import streamlit as st

from agent.graph import build_graph, run_agent

st.set_page_config(page_title="Zizo's Study Buddy", page_icon="🎓", layout="centered")

SUBJECTS = {
    "math": "Mathematics",
    "phyique": "Physics",
    "chimie": "Chemistry",
    "science": "Science",
    "technique": "Technology",
    "frensh": "French",
}


@st.cache_resource
def load_graph():
    """Build the graph once per process (it holds the SQLite checkpointer)."""
    return build_graph()


# --- Sidebar: subject + quiz score + new conversation -------------------
with st.sidebar:
    st.title("🎓 Zizo's Study Buddy")
    st.caption("Hi Zizo 👋 Ready to learn something new today?")
    subject_label = st.radio("Subject", list(SUBJECTS.values()),
                             label_visibility="collapsed")
    subject = next(k for k, v in SUBJECTS.items() if v == subject_label)

    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state[f"thread_{subject}"] = str(uuid.uuid4())
        st.session_state.pop(f"history_{subject}", None)
        st.session_state.pop(f"last_{subject}", None)
        st.rerun()

    last = st.session_state.get(f"last_{subject}")
    if last and (last["asked"] or last["score"]):
        st.divider()
        st.metric("Questions answered", last["asked"])
        st.metric("Correct answers", last["score"])

    st.divider()
    st.caption("Your personal study buddy — answers are based on your course material.")

# --- Thread state: one conversation per subject --------------------------
if f"thread_{subject}" not in st.session_state:
    st.session_state[f"thread_{subject}"] = f"ui-{subject}"
if f"history_{subject}" not in st.session_state:
    st.session_state[f"history_{subject}"] = []

thread_id = st.session_state[f"thread_{subject}"]
history = st.session_state[f"history_{subject}"]

st.subheader(subject_label)

# --- Transcript -----------------------------------------------------------
for i, msg in enumerate(history):
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("📖 Sources: " + " · ".join(msg["sources"]))

# --- Input + agent turn ----------------------------------------------------
if prompt := st.chat_input("Ask anything about your lesson..."):
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    graph = load_graph()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(graph, prompt, subject, thread_id)
        st.markdown(result["reply"])
        if result["sources"]:
            st.caption("📖 Sources: " + " · ".join(result["sources"]))

    history.append({"role": "assistant", "content": result["reply"],
                    "sources": result["sources"]})
    st.session_state[f"last_{subject}"] = result
    st.rerun()
