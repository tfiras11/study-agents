"""Shared LLM helpers, tuned for the CHAT_MODEL actually in use.

Two facts about nvidia/nemotron-3.5-lightning-30b-a3b (a REASONING model)
drive everything in this file:

1. It "thinks" before answering. The thinking goes to a separate
   `reasoning_content` field, but it COUNTS AGAINST max_tokens: with the
   library's small default budget the thinking eats everything and the
   visible answer comes back truncated -- sometimes nothing but the leaked
   reasoning. Every LLM we build here therefore gets a generous max_tokens.

2. Its hosted backend rejects `guided_json`, which is what
   `with_structured_output()` sends. So we do structured output the manual
   way instead: bind plain JSON mode (`response_format: json_object` --
   confirmed supported), tell the model the schema in the prompt, and
   validate the reply with Pydantic ourselves, retrying once on garbage.
"""
import json
import random
import time
from typing import Any, Protocol, Type, TypeVar

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from requests.exceptions import ConnectionError as RequestsConnectionError, ReadTimeout

from config import CHAT_MODEL, NVIDIA_API_KEY

T = TypeVar("T", bound=BaseModel)


class ChatInvoker(Protocol):
    """The common interface shared by ChatNVIDIA and its bound wrappers."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> BaseMessage:
        ...

# Room for thinking + answer. The thinking is NOT small: measured at up to
# ~4,000 tokens for a single quiz generation. With a 4096 budget the reply
# truncates mid-thought and the API mirrors the partial thinking into the
# visible content -- 8192 leaves real headroom for both.
DEFAULT_MAX_TOKENS = 8192

# A reasoning model sometimes spirals: the visible content comes back as
# its chain-of-thought (usually starting like this), often truncated or
# degenerate ("0. 0. 0..."). Such replies are garbage for the student --
# detect them and regenerate instead of showing them.
LEAK_PREFIXES = (
    "here's a thinking process",
    "here is a thinking process",
    "here's my thought process",
    "here's the",
    "here, the user",
    "here the user",
    "the user ",
    "let me think through",
    "okay,",
    "first, i",
)


def _looks_like_leak(content: Any) -> bool:
    text = content if isinstance(content, str) else _content_to_text(content)
    return text.lstrip().lower().startswith(LEAK_PREFIXES)


def make_llm(temperature: float = 0.3, max_tokens: int = DEFAULT_MAX_TOKENS) -> ChatNVIDIA:
    """One place to build the chat model, so every node shares the fix.

    timeout: the library default is 60s, which a reasoning model regularly
    blows through (thinking + answering). 300s gives it room.
    """
    return ChatNVIDIA(
        model=CHAT_MODEL,
        temperature=temperature,
        api_key=NVIDIA_API_KEY,
        max_tokens=max_tokens,
        timeout=300,
    )


def invoke_with_retry(llm: ChatInvoker, messages: list, max_retries: int = 3) -> BaseMessage:
    """llm.invoke() that rides out NVIDIA's transient failures.

    Two failure classes are retried: transport errors (the hosted endpoint
    regularly hangs -- ReadTimeout even at 300s -- or drops connections)
    and reasoning leaks (content = the model's chain-of-thought instead of
    the answer). One bad call shouldn't kill a student's turn.
    """
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages)
        except (ReadTimeout, RequestsConnectionError) as e:
            last_err = e
        else:
            if not _looks_like_leak(response.content):
                return response
            last_err = RuntimeError("raisonnement transféré dans la réponse visible")
        if attempt < max_retries:
            wait = 3 * attempt + random.uniform(0, 2)
            print(f"    ⚠️ réponse LLM inutilisable ({type(last_err).__name__}), "
                  f"nouvelle tentative dans {wait:.0f}s")
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def ask_structured(llm: ChatNVIDIA, schema: Type[T], messages: list,
                   max_attempts: int = 3) -> T:
    """Ask the model for a JSON object and validate it against `schema`.

    Replaces llm.with_structured_output(schema), which nemotron's backend
    rejects (400: unknown field `guided_json`). JSON mode guarantees the
    reply is parseable JSON; Pydantic guarantees it matches our schema.

    Nemotron quirks handled here:
    - it sometimes echoes the SCHEMA back instead of an instance, so the
      instruction leads with a concrete filled-in example of the format;
    - reasoning sometimes leaks into the content, burying the JSON inside
      thinking text -- we extract the first {...} block as a fallback;
    - on failure it gets up to `max_attempts` self-correction rounds where
      the retry message names the exact validation error.
    """
    skeleton = json.dumps(
        {name: f"<{info.description or name}>"
         for name, info in schema.model_fields.items()},
        ensure_ascii=False,
    )
    instruction = (
        "Réponds UNIQUEMENT avec un objet JSON valide (sans Markdown, sans "
        "commentaire), de ce format EXACT — un objet avec les champs remplis, "
        "PAS une description du format :\n"
        f"{skeleton}\n\n"
        "Détail des champs (schéma JSON) :\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
    json_llm = llm.bind(response_format={"type": "json_object"})
    convo = [SystemMessage(content=instruction)] + list(messages)

    feedback: tuple[str, str] | None = None
    last_error: ValidationError | None = None
    for _ in range(max_attempts):
        convo_i = convo if feedback is None else convo + [
            {"role": "assistant", "content": feedback[0]},
            {"role": "user", "content": feedback[1]},
        ]
        raw = _content_to_text(invoke_with_retry(json_llm, convo_i).content)
        text = _strip_fences(raw)

        # Try the whole text, then any {...} block buried in it (leaked
        # thinking sometimes surrounds the actual JSON object).
        candidates = [text]
        embedded = _first_json_object(text)
        if embedded is not None and embedded != text:
            candidates.append(embedded)

        for candidate in candidates:
            try:
                return schema.model_validate_json(candidate)
            except ValidationError as e:
                last_error = e

        feedback = (raw, (
            f"Ta réponse ne respecte pas le format demandé. Erreur : "
            f"{last_error}. ATTENTION : renvoie un objet JSON avec les "
            "champs REMPLIS de vraies valeurs (pas une description du "
            "format, pas le schéma, pas de raisonnement). Renvoie "
            "uniquement l'objet corrigé."
        ))

    raise last_error  # type: ignore[misc]


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} block in `text`, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _content_to_text(content: str | list[str | dict[Any, Any]]) -> str:
    """Extract text from either form LangChain uses for message content."""
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        else:
            raise TypeError(f"Unsupported model content block: {part!r}")
    return "".join(text_parts)


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` wrappers if the model adds them anyway."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t
