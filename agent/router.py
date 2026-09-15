"""The supervisor node: a tiny classifier that routes each message.

The supervisor does NOT tutor, quiz, or plan. It answers exactly one
question: "what does the student want?" -- and returns a route label.
That's why it runs at temperature 0: we want the same message to always
produce the same decision.

The key technique here is STRUCTURED OUTPUT: instead of parsing free text
and hoping, we hand the model a Pydantic schema and LangChain forces the
reply to fit that schema (under the hood it uses the model's function
calling / JSON mode). The result is a real Python object we can trust.
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from agent.llm import make_llm, ask_structured


class RouteDecision(BaseModel):
    """The only shape the router is allowed to answer with."""
    destination: Literal["tutor", "quiz", "planner", "chat"]
    reasoning: str = Field(description="One short sentence, for debugging.")


ROUTER_PROMPT = """You are the dispatcher of a study platform for a Tunisian
1ère année secondaire student. Read the student's last message and decide
where to send it:

- "tutor":   a question about a lesson, an exercise, a concept; the student
             wants an explanation or pastes a problem to solve.
- "quiz":    the student wants to be tested ("quiz me", "interroge-moi",
             "اختبرني", "donne-moi un exercice").
- "planner": the student asks what to study, how to organize revision, or
             wants a study plan.
- "chat":    anything else (greetings, thanks, small talk).

Choose the single best destination."""


def supervisor_node(state, config):
    """Classify the student's intent. Returns {"route": <destination>}."""
    llm = make_llm(temperature=0)
    decision = ask_structured(
        llm,
        RouteDecision,
        [
            SystemMessage(content=ROUTER_PROMPT),
            state["messages"][-1],  # the message being routed
        ],
    )
    print(f"[supervisor] -> {decision.destination} | {decision.reasoning}")
    return {"route": decision.destination}
