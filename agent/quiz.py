"""The quiz node: ONE node with TWO behaviors, chosen by the state.

This is a state machine living inside a LangGraph node:

    awaiting_answer == False  ->  PHASE 1 "generate":
        pick a lesson chunk, create a question + its expected answer,
        store both in state, flip awaiting_answer to True.

    awaiting_answer == True   ->  PHASE 2 "grade":
        the student's last message IS an answer. Grade it against
        expected_answer, update asked/score, generate the next question,
        and stay in "awaiting answer" mode.

Nobody ever "calls the quiz agent" directly: the STATE decides which
behavior runs. Note also what NEVER enters `messages`: expected_answer
and the bookkeeping fields travel in state (private side of the
whiteboard), so the model can never leak the answer to the student.
"""
import re

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage

from config import RETRIEVE_K
from agent.rag import retrieve, format_context
from agent.llm import make_llm, ask_structured


# --- Two contracts with the model --------------------------------------
class QuizQuestion(BaseModel):
    """What PHASE 1 must produce."""
    question: str = Field(description="One exercise, in simple French, with its page citation.")
    expected_answer: str = Field(description="The ideal answer. For grading only, never shown before the student answers.")


class GradeAndNext(BaseModel):
    """What PHASE 2 must produce: a verdict AND the next question, in one call."""
    correct: bool
    feedback: str = Field(description="2-3 sentences max: encourage first, then explain the mistake.")
    correct_answer: str = Field(description="The correct answer, revealed after grading.")
    next_question: str
    next_expected_answer: str


# --- PHASE 1 retrieval targeting -----------------------------------------
DEFAULT_QUIZ_QUERY = "exercices et applications du cours"

# Bare trigger phrases carry no topic info by themselves. Stripped out
# before checking whether the student actually named a topic, so "quiz me"
# still falls back to the generic query while "interroge-moi sur les
# fractions" retrieves fraction-related chunks instead of always pulling
# the same top-K exercises for the whole book.
_GENERIC_QUIZ_TRIGGERS = re.compile(
    r"\b(quiz[- ]?me|interroge[- ]?moi|teste[- ]?moi|"
    r"donne[- ]?moi\s+un\s+exercice|اختبرني)\b",
    re.IGNORECASE,
)


def _quiz_topic_query(student_message: str) -> str:
    """Build the PHASE 1 retrieval query from the student's message.

    Biases retrieval toward a named topic when there is one; otherwise
    falls back to the generic exercise-flavoured query so a bare "quiz me"
    still retrieves something reasonable.
    """
    residual = _GENERIC_QUIZ_TRIGGERS.sub("", student_message).strip(" ,.!?:;-")
    if len(residual) < 3:  # nothing left once the trigger phrase is stripped
        return DEFAULT_QUIZ_QUERY
    return f"{residual} — {DEFAULT_QUIZ_QUERY}"


GENERATE_PROMPT = """Tu prépares un mini-quiz pour un élève de 1ère année
secondaire (15-16 ans). Génère UNE seule question d'exercice, en français
simple, STRICTEMENT à partir du contexte de cours ci-dessous (comme en
classe : application directe du cours, pas de hors-programme). Cite la page
entre parenthèses dans la question. Format libre : QCM ou réponse courte.

CONTEXTE DE COURS :
{context}"""

GRADE_PROMPT = """Tu corriges la réponse d'un élève de 1ère année secondaire,
de façon bienveillante. Utilise le contexte de cours pour juger.

CONTEXTE DE COURS :
{context}

QUESTION POSÉE :
{question}

RÉPONSE ATTENDUE (référence) :
{expected}

RÉPONSE DE L'ÉLÈVE :
{student}

Consignes : accepte les réponses justes même si elles sont mal formulées ou
contiennent une faute d'orthographe ; note correct=false seulement si le
fond est faux. Feedback en 2-3 phrases : commence par encourager, puis
explique l'erreur. Ensuite génère la question SUIVANTE, même règles que la
précédente (issue du contexte, page citée)."""


def quiz_node(state, config):
    subject = config["configurable"]["subject"]
    llm = make_llm(temperature=0.3)

    # ---------------- PHASE 1: no question on the table -> generate one --
    if not state.get("awaiting_answer"):
        topic_query = _quiz_topic_query(state["messages"][-1].content)
        docs = retrieve(subject, topic_query, k=RETRIEVE_K)
        if not docs:
            # Graceful degradation: no ingested material, no fake quiz.
            return {
                "messages": [AIMessage(content=(
                    "Je n'ai pas encore ton cours en mémoire pour cette "
                    "matière, donc je ne peux pas te faire un quiz fidèle "
                    "à ta leçon. Demande d'abord d'ajouter le chapitre, "
                    "puis reviens ! 💪"
                ))],
            }

        quiz = ask_structured(
            llm, QuizQuestion,
            [SystemMessage(content=GENERATE_PROMPT.format(context=format_context(docs)))],
        )
        announce = AIMessage(
            content=f"📝 **Question {state.get('asked', 0) + 1}** :\n\n{quiz.question}"
        )
        return {
            "messages": [announce],        # appended (public side)
            "awaiting_answer": True,       # flip the state machine
            "current_question": quiz.question,
            "expected_answer": quiz.expected_answer,  # private side: never in messages
        }

    # ---------------- PHASE 2: a question is on the table -> grade --------
    student_answer = state["messages"][-1].content
    docs = retrieve(subject, state["current_question"], k=RETRIEVE_K)

    result = ask_structured(
        llm, GradeAndNext,
        [SystemMessage(content=GRADE_PROMPT.format(
            context=format_context(docs),
            question=state["current_question"],
            expected=state["expected_answer"],
            student=student_answer,
        ))],
    )

    asked = state.get("asked", 0) + 1
    score = state.get("score", 0) + (1 if result.correct else 0)

    verdict = "✅ **Correct !**" if result.correct else "❌ **Pas tout à fait...**"
    content = (
        f"{verdict}\n\n{result.feedback}\n\n"
        f"💡 Bonne réponse : {result.correct_answer}\n\n---\n\n"
        f"📝 **Question {asked + 1}** :\n\n{result.next_question}"
    )
    return {
        "messages": [AIMessage(content=content)],
        "awaiting_answer": True,                       # quiz continues
        "current_question": result.next_question,      # slide the window:
        "expected_answer": result.next_expected_answer,  # next becomes current
        "asked": asked,
        "score": score,
    }