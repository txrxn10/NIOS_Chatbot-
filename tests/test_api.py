from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_unauthenticated_personal_query_blocked():
    response = client.post(
        "/api/v1/chat/query",
        json={"query": "Show my exam marks and result"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "please sign in" in data["answer"].lower()
    assert data["sources"] == []


def test_authenticated_module_1_admission_status():
    response = client.post(
        "/api/v1/chat/query",
        headers={"Authorization": "Bearer dev_student_270041123456"},
        json={"query": "What is my admission status?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Taran Mahajan" in data["answer"]
    assert "270041123456" in data["answer"]


def test_authenticated_module_3_results():
    response = client.post(
        "/api/v1/chat/query",
        headers={"Authorization": "Bearer dev_student_270041123456"},
        json={"query": "Show my marks and result status"}
    )
    assert response.status_code == 200
    assert "PASSED" in response.json()["answer"]