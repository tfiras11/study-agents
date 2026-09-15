# Study Agents

An AI study companion for Tunisian first-year secondary-school students. The
Streamlit interface offers subject-specific conversations in French, grounded
in the student's course material rather than general-purpose answers.

## What it does

- Answers lesson and exercise questions using retrieval-augmented generation
  (RAG) over the course content.
- Generates and grades a continuing, source-grounded mini-quiz.
- Creates revision plans from the subject syllabus.
- Keeps a separate conversation and score for each subject.
- Routes each message through a LangGraph supervisor to tutor, quiz, planner,
  or friendly-chat nodes.

Supported subjects: mathematics, physics, chemistry, science, technology, and
French.

## Architecture

```text
Student -> Streamlit -> LangGraph supervisor
                         |-> tutor   -> Chroma retrieval -> NVIDIA LLM
                         |-> quiz    -> Chroma retrieval -> NVIDIA LLM
                         |-> planner -> syllabus file   -> NVIDIA LLM
                         `-> chat    -> NVIDIA LLM
```

Course PDFs can be ingested with a vision model. Their pages are transcribed to
Markdown, chunked with page metadata, embedded locally with a multilingual
embedding model, and persisted in Chroma. The checked-in `db/chroma` folder
contains the prepared vector database so the app can answer from existing
chunks without re-ingestion.

## Quick start

Prerequisites: Python 3.11+ and an NVIDIA API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `NVIDIA_API_KEY` in `.env`, then start the interface:

```powershell
streamlit run app.py
```

## Adding course material

Place a PDF locally and ingest it into a subject collection:

```powershell
python ingest.py "data/textbooks/math_ch3.pdf" --subject math
```

Useful options:

```text
--skip-pages N    Ignore cover/table-of-contents pages.
--page-offset N   Align PDF page numbers with printed textbook page numbers.
```

Ingestion uses the configured NVIDIA vision model to preserve mathematical
notation and Arabic text, then upserts the resulting chunks into Chroma.

## Project layout

```text
app.py              Streamlit chat application
agent/               LangGraph nodes, state, routing, RAG, and LLM helpers
config.py            Paths, models, retrieval settings, and environment config
ingest.py            PDF-to-Chroma ingestion pipeline
data/syllabus/       Subject chapter lists used by the planner
db/chroma/           Versioned Chroma vector database
db/checkpoints.sqlite Local conversation history (ignored)
```

## Configuration

`config.py` centralizes model names, chunking settings, and storage locations.
The API key is read only from `NVIDIA_API_KEY` in `.env`; never commit that
file. The original textbooks, generated transcriptions, temporary images,
virtual environment, and local chat checkpoint are ignored by Git.

## Deployment note

`agent.graph:graph` is the deployment export and is compiled without the local
SQLite checkpointer, allowing a managed LangGraph/LangSmith deployment to
provide persistence. The Streamlit app uses `build_graph()` for local SQLite
conversation memory.

## License and content

Before making this repository public, ensure you have permission to distribute
any course material represented by the vector database. Original textbook PDFs
are intentionally not tracked.
