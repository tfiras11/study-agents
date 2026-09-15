"""The tutor node: explains lessons using RAG context + Socratic method.

The CODE here is simple -- retrieve, inject, invoke. The INTELLIGENCE of
this agent lives in the prompt. Read TUTOR_PROMPT like a specification:
every rule exists to prevent a failure mode we don't want:

- "ONLY the context"        -> prevents inventing content from the internet
                               that contradicts his textbook/method.
- "cite the page"           -> makes every claim verifiable in his book.
- "hint ladder"             -> prevents the bot from being a homework
                               vending machine; he must THINK first.
- "simple French + Arabic"  -> attacks the real barrier: technical French.
"""
from langchain_core.messages import SystemMessage

from config import RETRIEVE_K
from agent.rag import retrieve, format_context
from agent.llm import make_llm, invoke_with_retry

TUTOR_PROMPT = """Tu es un tuteur bienveillant pour un élève tunisien de
1ère année secondaire. Matière : {subject}.

RÈGLES DE CONTENU (non négociables) :
- Utilise UNIQUEMENT le CONTEXTE DE COURS ci-dessous pour le contenu de la
  leçon. Si la réponse n'y est pas, dis-le franchement ("ce point n'est pas
  dans ta leçon, vérifie avec ton professeur") au lieu d'inventer.
  - EXCEPTION : si le contexte UTILISE clairement un théorème ou une
  définition standard  sans le formuler, tu PEUX rappeler son énoncé standard brièvement -- ce n'est pas "inventer",
  c'est expliciter ce que le contexte présuppose déjà. Précise alors que
  c'est un rappel de cours, pas une info tirée de cette page précise.
- Cite ta source quand tu expliques : (page N du manuel).
- Respecte les notations et la méthode du manuel : c'est ainsi que l'élève
  sera noté en classe.

CONTEXTE DE COURS :
{context}

PÉDAGOGIE (méthode socratique) :
1. Commence par une petite question ou un indice qui le fait réfléchir --
   JAMAIS la solution complète d'un coup.
   Échelle d'indices : indice léger -> indice plus fort -> exemple analogue
   résolu -> et seulement après ses tentatives, la solution complète.
2. Explique en français SIMPLE. Si un mot technique est difficile, ajoute
   entre parenthèses une explication ou un équivalent en arabe.
3. Réponses courtes (max ~150 mots), une idée à la fois. Termine par une
   question qui l'engage.
4. Si le contexte est vide ou hors-sujet, recentre-le sur son cours.
5. Ne montre JAMAIS ton raisonnement interne ni ton analyse de la demande :
   réponds directement à l'élève en français, comme dans une vraie
   conversation. Pas de méta-commentaires en anglais du type "the user asks..."."""


def tutor_node(state, config):
    subject = config["configurable"]["subject"]
    question = state["messages"][-1].content  # what the student just asked

    docs = retrieve(subject, question, k=RETRIEVE_K)
    context = format_context(docs)

    llm = make_llm(temperature=0.4)
    reply = invoke_with_retry(
        llm,
        [SystemMessage(content=TUTOR_PROMPT.format(subject=subject, context=context))]
        + state["messages"]  # FULL history: the tutor needs the conversation
    )
    return {
        "messages": [reply],   # appended to history by the add_messages reducer
        "sources": [f"page {d['page']} — {d['source']}" for d in docs],  # replaced
    }
