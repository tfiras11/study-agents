"""The AgentState: one shared "whiteboard" that all our agents read and write.

In LangGraph there is no hidden magic: the graph passes this dict from node
to node. Every node receives the whole state, and returns ONLY the fields
it wants to update. LangGraph merges those updates using the "reducer"
declared in each field's Annotated type.

Two merge rules to understand:

- `messages` uses the add_messages reducer  -> new messages are APPENDED.
- every other field has no reducer          -> the returned value REPLACES
  the old one (last writer wins).
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history. Appended on every turn by the add_messages
    # reducer -- that is how the bot "remembers" the conversation.
    messages: Annotated[list, add_messages]

    # Written by the supervisor (router), read by the conditional edge
    # that decides which agent node runs next.
    route: Literal["tutor", "quiz", "planner", "chat"]

    # Page citations from the last RAG-backed answer (shown in the UI).
    sources: list[str]

    # --- Quiz state machine (owned by the quiz node) --------------------
    awaiting_answer: bool   # True = a question is on the table; the next
                            # student message IS an answer, not a request.
    current_question: str   # the question currently asked
    expected_answer: str    # what a correct answer looks like (for grading)
    asked: int              # questions asked so far in this session
    score: int              # correct answers so far
