from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ask_rejects_empty_question():
    res = client.post("/ask", json={"q": "   ", "doc_id": "anything"})
    assert res.status_code == 400
    assert res.json()["detail"] == "Question cannot be empty"


def test_ask_rejects_empty_doc_id():
    res = client.post("/ask", json={"q": "hello", "doc_id": "   "})
    assert res.status_code == 400
    assert res.json()["detail"] == "Document id is required"


def test_ask_bad_doc_id_returns_404():
    res = client.post("/ask", json={"q": "hello", "doc_id": "does-not-exist"})
    assert res.status_code == 404
    assert "Document not found" in res.json()["detail"]


def test_upload_rejects_non_pdf_by_filename():
    # We only test the filename validator (no need for real PDF parsing here).
    res = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Only PDF files are allowed"

