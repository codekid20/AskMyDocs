"""FastAPI backend: wraps the per-session RAG pipeline in HTTP endpoints.

Each browser session gets a unique session_id -> its own Chroma collection, so
uploaded documents are isolated per session. Endpoints:
  POST /session         -> create a session, returns {session_id}
  POST /upload          -> ingest a PDF into the session's collection
  POST /chat            -> ask a question, get a grounded answer + citations
  DELETE /session/{id}  -> tear down a session's collection
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from askmydocs.embed import get_store
from askmydocs.generate import answer_question
from askmydocs.upload import ingest_pdf

app = FastAPI(title="Ask My Docs")

# Allow the React dev server (localhost:5173 for Vite) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track which docs are in each session (for the UI to show what's uploaded).
_sessions: dict[str, list[str]] = {}


def _collection(session_id: str) -> str:
    # Chroma collection names have charset rules; a uuid hex is safe.
    return f"session_{session_id}"


class SessionOut(BaseModel):
    session_id: str


class ChatIn(BaseModel):
    session_id: str
    question: str


class Citation(BaseModel):
    n: int
    source: str


class ChatOut(BaseModel):
    answer: str
    citations: list[Citation]


@app.post("/session", response_model=SessionOut)
def create_session() -> SessionOut:
    sid = uuid.uuid4().hex
    _sessions[sid] = []
    logger.info("Created session {}", sid)
    return SessionOut(session_id=sid)


@app.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    if session_id not in _sessions:
        raise HTTPException(404, "Unknown session — create one first.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Write the upload to a temp file (PyMuPDF needs a path on disk).
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        n = ingest_pdf(tmp_path, _collection(session_id))
    finally:
        tmp_path.unlink(missing_ok=True)

    _sessions[session_id].append(file.filename)
    return {"filename": file.filename, "chunks_indexed": n,
            "documents": _sessions[session_id]}


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    if body.session_id not in _sessions:
        raise HTTPException(404, "Unknown session — create one first.")
    if not _sessions[body.session_id]:
        raise HTTPException(400, "No documents uploaded for this session yet.")

    ans = answer_question(body.question, collection=_collection(body.session_id))
    return ChatOut(
        answer=ans.text,
        citations=[Citation(n=n, source=c.source.display()) for n, c in ans.cited],
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    if session_id in _sessions:
        try:
            get_store(_collection(session_id)).delete_collection()
        except Exception as e:  # noqa: BLE001
            logger.warning("Collection teardown failed: {}", e)
        _sessions.pop(session_id, None)
    return {"deleted": session_id}