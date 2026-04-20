from app.store import save_document, list_documents, find_document_by_hash, delete_document
from app.rag import prepare_vector_store, load_vector_store, ask_question
from app.pdf_loader import load_pdf
import shutil
import uuid
import os
import hashlib
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag-project")


app = FastAPI()

_rate_counters = {}


def _check_rate_limit(ip: str, route_key: str, limit: int, window_seconds: int) -> None:
    """
    Very simple in-memory rate limit.
    Stores request timestamps per (ip, route_key) and rejects if too many occur within window.
    """
    now = time.time()
    key = (ip or "unknown", route_key)
    timestamps = _rate_counters.get(key, [])

    cutoff = now - window_seconds
    timestamps = [t for t in timestamps if t >= cutoff]

    if len(timestamps) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Try again later. ({limit}/{window_seconds}s)",
        )

    timestamps.append(now)
    _rate_counters[key] = timestamps

cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if not cors_origins_env:
    raise RuntimeError(
        "CORS_ORIGINS is missing. Add it to your .env (comma-separated origins)."
    )

cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    _check_rate_limit(request.client.host if request.client else "unknown", "upload", limit=5, window_seconds=60)
    t0 = time.perf_counter()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.info("upload rejected request_id=%s ip=%s filename=%s", request_id, request.client.host if request.client else "unknown", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Ensure storage folder exists
    os.makedirs("data", exist_ok=True)

    safe_name = os.path.basename(file.filename)

    # Learning-friendly safety limit.
    # Prevents huge uploads from causing very slow processing or unexpected cost.
    MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

    # Save upload to a temporary file first so we can hash it.
    temp_path = f"data/upload_{uuid.uuid4()}.pdf"
    file_hash = hashlib.sha256()
    total_bytes = 0
    t_save_start = time.perf_counter()
    with open(temp_path, "wb") as buffer:
        while True:
            chunk = file.file.read(8192)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                # Cleanup temp file before returning error
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail="File is too large. Please upload a PDF under 20 MB.",
                )
            buffer.write(chunk)
            file_hash.update(chunk)
    t_save_ms = (time.perf_counter() - t_save_start) * 1000

    upload_hash = file_hash.hexdigest()

    # If we've already seen this exact PDF (same bytes), reuse it.
    existing = find_document_by_hash(upload_hash)
    if existing:
        try:
            os.remove(temp_path)
        except OSError:
            pass

        logger.info(
            "upload dedup request_id=%s ip=%s doc_id=%s name=%s bytes=%s save_ms=%.2f total_ms=%.2f",
            request_id,
            request.client.host if request.client else "unknown",
            existing["doc_id"],
            existing["name"],
            total_bytes,
            t_save_ms,
            (time.perf_counter() - t0) * 1000,
        )

        return {"doc_id": existing["doc_id"], "name": existing["name"], "request_id": request_id}

    doc_id = str(uuid.uuid4())
    file_path = f"data/{doc_id}.pdf"
    os.replace(temp_path, file_path)

    try:
        # Process PDF
        t_pdf_start = time.perf_counter()
        docs = load_pdf(file_path)
        t_pdf_ms = (time.perf_counter() - t_pdf_start) * 1000

        t_index_start = time.perf_counter()
        prepare_vector_store(docs, doc_id)
        t_index_ms = (time.perf_counter() - t_index_start) * 1000
    except Exception as e:
        # Cleanup partial artifacts so /documents and /ask don't break later
        try:
            os.remove(file_path)
        except OSError:
            pass

        try:
            shutil.rmtree(f"data/faiss_{doc_id}", ignore_errors=True)
        except OSError:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}",
        )

    # Save metadata only after processing succeeds
    save_document(safe_name, doc_id, file_hash=upload_hash)

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "upload ok request_id=%s ip=%s doc_id=%s name=%s bytes=%s save_ms=%.2f pdf_load_ms=%.2f index_ms=%.2f total_ms=%.2f",
        request_id,
        request.client.host if request.client else "unknown",
        doc_id,
        safe_name,
        total_bytes,
        t_save_ms,
        t_pdf_ms,
        t_index_ms,
        total_ms,
    )
    return {
        "doc_id": doc_id,
        "name": safe_name,
        "request_id": request_id,
        "metrics": {
            "bytes": total_bytes,
            "save_ms": round(t_save_ms, 2),
            "pdf_load_ms": round(t_pdf_ms, 2),
            "index_ms": round(t_index_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    }


@app.get("/documents")
def get_documents():
    return list_documents()


@app.delete("/documents/{doc_id}")
def delete_doc(request: Request, doc_id: str):
    request_id = str(uuid.uuid4())
    doc_id = doc_id.strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="Document id is required")

    # Remove disk artifacts first (safe if missing)
    try:
        os.remove(f"data/{doc_id}.pdf")
    except OSError:
        pass

    try:
        shutil.rmtree(f"data/faiss_{doc_id}", ignore_errors=True)
    except OSError:
        pass

    deleted = delete_document(doc_id)
    if not deleted:
        logger.info("delete not_found request_id=%s ip=%s doc_id=%s", request_id, request.client.host if request.client else "unknown", doc_id)
        raise HTTPException(status_code=404, detail="Document not found")

    logger.info("delete ok request_id=%s ip=%s doc_id=%s", request_id, request.client.host if request.client else "unknown", doc_id)
    return {"deleted": True, "doc_id": doc_id}


class AskRequest(BaseModel):
    q: str
    doc_id: str


@app.post("/ask")
def ask(request: Request, payload: AskRequest):
    request_id = str(uuid.uuid4())
    _check_rate_limit(request.client.host if request.client else "unknown", "ask", limit=30, window_seconds=60)
    if not payload.q.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not payload.doc_id.strip():
        raise HTTPException(status_code=400, detail="Document id is required")

    try:
        if not os.path.exists(f"data/faiss_{payload.doc_id}"):
            raise HTTPException(
                status_code=404,
                detail="Document not found or has not been processed correctly",
            )

        vector_store = load_vector_store(payload.doc_id)
        result = ask_question(vector_store, payload.q)
    except HTTPException:
        raise
    except Exception as e:
        logger.info(
            "ask error request_id=%s ip=%s doc_id=%s q_len=%s err=%s",
            request_id,
            request.client.host if request.client else "unknown",
            payload.doc_id,
            len(payload.q),
            str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to answer question. Check server logs.")

    # Log useful debug metrics from RAG stage if present
    retrieval = result.get("retrieval") if isinstance(result, dict) else None
    logger.info(
        "ask ok request_id=%s ip=%s doc_id=%s q_len=%s weak=%s retrieval_ms=%s llm_ms=%s total_ms=%s",
        request_id,
        request.client.host if request.client else "unknown",
        payload.doc_id,
        len(payload.q),
        retrieval.get("weak_retrieval") if isinstance(retrieval, dict) else None,
        retrieval.get("retrieval_ms") if isinstance(retrieval, dict) else None,
        retrieval.get("llm_ms") if isinstance(retrieval, dict) else None,
        retrieval.get("total_ms") if isinstance(retrieval, dict) else None,
    )

    if isinstance(result, dict):
        result["request_id"] = request_id
    return result
