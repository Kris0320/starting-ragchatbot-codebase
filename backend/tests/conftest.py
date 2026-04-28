"""
Shared fixtures and test app for RAG system API tests.

app.py mounts static files from ../frontend and initialises RAGSystem at import
time (heavy: ChromaDB + sentence-transformers). To keep tests fast and
self-contained, this module builds a minimal FastAPI app that mirrors the same
routes but accepts an injected mock, so no real infrastructure is needed.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Pydantic models (mirrors app.py — kept here so tests never import app.py)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------

def create_test_app(rag_system) -> FastAPI:
    """Return a FastAPI app wired to *rag_system* instead of the real one.

    Deliberately omits the static-file mount so tests run without a frontend
    build present. A plain JSON root route stands in for the HTML shell.
    """
    app = FastAPI(title="RAG System — Test")

    @app.get("/")
    async def root():
        return {"status": "ok"}

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """Pre-configured MagicMock that satisfies all RAGSystem call sites."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    rag.query.return_value = ("Test answer", ["source1.txt"])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag


@pytest.fixture
def client(mock_rag_system):
    """Synchronous TestClient backed by the test app and a mock RAG system."""
    app = create_test_app(mock_rag_system)
    with TestClient(app) as test_client:
        yield test_client
