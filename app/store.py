import uuid
import json
import os

DATA_FILE = "data/documents.json"


def load_documents():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_documents(documents):
    with open(DATA_FILE, "w") as f:
        json.dump(documents, f, indent=2)

def find_document_by_hash(file_hash):
    documents = load_documents()
    for doc in documents:
        if isinstance(doc, dict) and doc.get("file_hash") == file_hash:
            return doc
    return None


def save_document(name, doc_id=None, file_hash=None):
    documents = load_documents()
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    doc = {
        "doc_id": doc_id,
        "name": name,
    }

    if file_hash:
        doc["file_hash"] = file_hash

    documents.append(doc)
    save_documents(documents)
    return doc_id


def list_documents():
    documents = load_documents()
    # Learning-friendly cleanup: only show documents that still have a saved vector store.
    # This prevents old/broken entries (e.g., non-PDF uploads) from appearing in the UI.
    valid_documents = []
    for doc in documents:
        doc_id = doc.get("doc_id") if isinstance(doc, dict) else None
        if not doc_id:
            continue

        if os.path.exists(f"data/faiss_{doc_id}"):
            valid_documents.append(doc)

    return valid_documents[::-1]


def delete_document(doc_id: str) -> bool:
    documents = load_documents()
    new_docs = []
    deleted = False

    for doc in documents:
        if isinstance(doc, dict) and doc.get("doc_id") == doc_id:
            deleted = True
            continue
        new_docs.append(doc)

    if deleted:
        save_documents(new_docs)
    return deleted
