"""API endpoint tests for the RAG chatbot.

Uses the test app and fixtures from conftest.py — no real ChromaDB, embeddings,
or Anthropic API calls are made during these tests.
"""


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_new_session_is_created_when_none_provided(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What is machine learning?"})

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_1"
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["source1.txt"]
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_existing_session_id_is_reused(self, client, mock_rag_system):
        response = client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "session_existing"},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "session_existing"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_empty_query_string_is_accepted(self, client):
        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200

    def test_rag_error_returns_500_with_detail(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("Vector store unavailable")

        response = client.post("/api/query", json={"query": "What is deep learning?"})

        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]

    def test_response_with_no_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("No relevant content found.", [])

        response = client.post("/api/query", json={"query": "An obscure topic"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_rag_query_is_called_with_correct_arguments(self, client, mock_rag_system):
        client.post(
            "/api/query",
            json={"query": "Explain neural networks", "session_id": "s42"},
        )

        mock_rag_system.query.assert_called_once_with("Explain neural networks", "s42")


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    def test_returns_course_count_and_titles(self, client):
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course A", "Course B"]

    def test_empty_catalog_returns_zero_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_error_returns_500_with_detail(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB connection failed")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "DB connection failed" in response.json()["detail"]

    def test_response_shape_matches_schema(self, client):
        response = client.get("/api/courses")

        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
