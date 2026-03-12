# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8000 --loop asyncio

# Run with auto-reload (development)
uvicorn main:app --reload --loop asyncio

# Build Docker image
docker build -t godata-backend .

# Run Docker container
docker run -p 8000:8000 --env-file .env godata-backend
```

There are no test commands — the project has no test suite.

## Architecture

This is a Python **FastAPI** backend that powers a bilingual (English/Spanish) AI chatbot for GoData. It uses a RAG (Retrieval-Augmented Generation) pipeline backed by LangGraph agent orchestration.

### Request Flow

`POST /api/chat` → `main.py` → `Brain.ask()` in `rag/rag.py` → LangGraph agent (with tools) → OpenAI `gpt-4o-mini`

### Startup Initialization (`main.py`)

The `Brain` initializes asynchronously in the background during server startup:
1. Playwright scrapes `TARGET_URL` (all internal pages)
2. Text is chunked (1000 chars, 200 overlap) and embedded with HuggingFace `all-MiniLM-L6-v2`
3. Embeddings stored in an in-memory Chroma vector database
4. LangGraph agent is created with tools bound

The `/` health endpoint exposes `brain_ready: bool` so callers can wait for initialization.

### RAG & Agent (`rag/rag.py`)

- **LLM**: `ChatOpenAI` with `gpt-4o-mini`, `temperature=0`
- **Agent**: LangGraph `create_react_agent` with `MemorySaver` (thread-based conversation history keyed by `thread_id`)
- **Short-query optimization**: Queries of ≤3 words skip vector DB retrieval and go directly to the LLM
- **Context injection**: Retrieved chunks are prepended to the user message before sending to the agent
- **System prompt persona**: "Jose", GoData AI Architect and support specialist

### Tools (`rag/tools.py`)

Three LangChain tools the agent can invoke:

| Tool | Purpose |
|------|---------|
| `validate_email_format` | Regex + MX record lookup via DNS |
| `check_team_availability` | Microsoft Graph Calendar API — checks Juan Montiel's 30-min window |
| `book_godata_meeting` | Microsoft Graph API — creates Teams meeting, adds user + Juan Montiel as attendees |

Meeting bookings always use **Ecuador time (SA Pacific Standard Time, UTC-5)** and always include `juan.montiel@godata.com.ec` as organizer/attendee.

### Environment Variables Required

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API access |
| `TARGET_URL` | Website URL to scrape for RAG knowledge base |
| `GODATA_BOOKINGS_LINK` | Outlook booking link (informational) |
| `MS_CLIENT_ID` | Azure AD app registration client ID |
| `MS_TENANT_ID` | Azure AD tenant ID |
| `MS_CLIENT_SECRET` | Azure AD app client secret |

### Deployment

CI/CD is via `.github/workflows/main_godataec-ai-bk.yml`: push to `main` builds a Docker image, pushes to Docker Hub (`camd0204/godata-backend`), and deploys to Azure Web App (`godata-bak-ai`).
