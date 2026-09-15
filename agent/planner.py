"""The planner node: builds a realistic revision plan.

New pattern here: grounding with a LOCAL FILE. The syllabus
(data/syllabus/{subject}.md -- the chapter list you paste from the official
program) is small enough to fit entirely in the prompt, so it needs NO
vector store, NO retrieval. Rule of thumb: RAG is for documents too big to
paste; for small ones, just paste them.

Also new: injecting RUNTIME context (today's date). The model has no clock;
without it, "plan my week" has no anchor.
"""
from datetime import date
from pathlib import Path

from langchain_core.messages import SystemMessage

from config import SYLLABUS_DIR
from agent.llm import make_llm, invoke_with_retry

PLANNER_PROMPT = """Tu es un coach d'étude pour un élève tunisien de 1ère
année secondaire. Matière : {subject}. Aujourd'hui : {today}.

PROGRAMME DU COURS (chapters officiels) :
{syllabus}

DEMANDE DE L'ÉLÈVE (ses difficultés, ce qu'il veut réviser) :
{concerns}

Construis un plan de révision de 7 jours :
- 90 à 120 minutes par jour, UNE seule chose par jour (il a d'autres devoirs).
- Priorise ses points faibles; alterne revoir (ce qui est déjà vu en classe)
  et approfondir.
- Replanifie chaque point faible 3 jours plus tard (répétition espacée).
- Ancre chaque jour sur un chapitre DU PROGRAMME ci-dessus, pas d'invention.
- Termine par un conseil d'organisation en une phrase.
- Français simple, plan clair jour par jour (Jour 1, Jour 2, ...)."""


def planner_node(state, config):
    subject = config["configurable"]["subject"]

    # The syllabus is a small file -> read it whole, no RAG needed.
    syllabus_path = SYLLABUS_DIR / f"{subject}.md"
    if syllabus_path.exists():
        # utf-8 matters: the syllabus may contain Arabic, and Windows'
        # default encoding would mangle it silently.
        syllabus = syllabus_path.read_text(encoding="utf-8")
    else:
        syllabus = ("(programme non chargé — base-toi sur le programme "
                    "standard tunisien de 1ère année secondaire)")

    concerns = state["messages"][-1].content
    today = date.today().strftime("%A %d %B %Y")

    llm = make_llm(temperature=0.5)
    reply = invoke_with_retry(
        llm,
        [SystemMessage(content=PLANNER_PROMPT.format(
            subject=subject, today=today, syllabus=syllabus, concerns=concerns,
        ))]
        + state["messages"][-6:]  # recent context is enough for planning
    )
    return {
        "messages": [reply],
        "sources": [],  # not RAG-backed -> clear stale page citations
    }
