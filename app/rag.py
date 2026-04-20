from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time
import os


def prepare_vector_store(docs, doc_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 💾 Save to disk
    vector_store.save_local(f"data/faiss_{doc_id}")

    return vector_store


def load_vector_store(doc_id):
    embeddings = OpenAIEmbeddings()

    return FAISS.load_local(
        f"data/faiss_{doc_id}",
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_question(vector_store, query):
    t0 = time.perf_counter()
    # Retrieve top-k chunks so we can show sources later (and also scores)
    t_retrieval_start = time.perf_counter()
    docs_and_scores = vector_store.similarity_search_with_score(query, k=6)
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000
    docs = [doc for doc, _score in docs_and_scores]
    # FAISS returns numpy float types sometimes; convert early to plain Python float.
    scores = [float(score) for _doc, score in docs_and_scores]

    # Learning-friendly guardrail: keep LLM context reasonably small.
    # (Otherwise large PDFs can cause token/cost spikes or failures.)
    MAX_CHARS_PER_EXCERPT = 1800

    def display_page(doc):
        page = (doc.metadata or {}).get("page")
        if isinstance(page, int):
            # PyPDFLoader pages are typically 0-indexed
            return page + 1
        if isinstance(page, str) and page.isdigit():
            return int(page) + 1
        return None

    # Number excerpts only (no page headers in prompt text — avoids "Page N" in answers).
    context_parts = []
    total_chars = 0
    for i, doc_and_score in enumerate(docs_and_scores, start=1):
        doc, score = doc_and_score
        content = doc.page_content or ""
        truncated = len(content) > MAX_CHARS_PER_EXCERPT
        content = content[:MAX_CHARS_PER_EXCERPT]
        total_chars += len(content)

        suffix = " ...(truncated)" if truncated else ""
        # Score is a retrieval metric from FAISS; lower usually means more similar (distance).
        context_parts.append(f"[Excerpt {i}]\n{content}{suffix}")
    context = "\n\n".join(context_parts)

    # For FAISS distance metrics: lower scores are better.
    min_score = min(scores) if scores else None
    avg_score = (sum(scores) / len(scores)) if scores else None
    spread_ratio = ((avg_score - min_score) / avg_score) if avg_score not in (None, 0) else 0

    # Weak retrieval heuristic (tunable):
    # "Spread-only" checks are often too aggressive for resumes, where top chunks can look similarly relevant.
    # So we only treat retrieval as weak when BOTH:
    # - the top matches are relatively far (high distance), AND
    # - the matches are unusually close to each other (low spread).
    spread_threshold_env = os.getenv("WEAK_RETRIEVAL_SPREAD_THRESHOLD", "").strip()
    spread_threshold = float(spread_threshold_env) if spread_threshold_env else 0.03

    min_avg_env = os.getenv("WEAK_RETRIEVAL_MIN_AVG_SCORE", "").strip()
    min_avg = float(min_avg_env) if min_avg_env else 0.62

    min_best_env = os.getenv("WEAK_RETRIEVAL_MIN_BEST_SCORE", "").strip()
    min_best = float(min_best_env) if min_best_env else 0.62

    weak_retrieval = False
    if (
        min_score is not None
        and avg_score is not None
        and spread_ratio < spread_threshold
        and avg_score >= min_avg
        and min_score >= min_best
    ):
        weak_retrieval = True

    # Optional absolute threshold (configurable via env var).
    # If set, and the best distance is still "too high", we treat retrieval as weak.
    max_distance_env = os.getenv("FAISS_MAX_DISTANCE", "").strip()
    max_distance = float(max_distance_env) if max_distance_env else None
    max_distance_triggered = False
    if max_distance is not None and min_score is not None and min_score > max_distance:
        weak_retrieval = True
        max_distance_triggered = True
    if weak_retrieval:
        context = ""

    prompt = f"""
You are answering using ONLY the provided context excerpts.

Rules:
1. If the excerpts are empty, say exactly:
   I don't know based on the provided document.
2. Do not use outside knowledge.
3. If the question asks for a summary/overview, write a short grounded summary using only supported facts from the excerpts.
   Begin with a short label for what the excerpts clearly are when that is obvious from the text alone
   (e.g. "This job posting ...", "This resume/CV ...", "This cover letter ...").
   Do not call it a resume or CV unless the excerpts clearly show first-person career/profile content
   (work history, skills, education, contact) typical of a resume—not a third-person job ad.
   If the kind of document is unclear, say "This document ..." or "The uploaded file ...".
4. If the excerpts do not contain enough information to answer, say exactly:
   I don't know based on the provided document.
5. Do not mention page numbers, excerpt numbers, or "Page N" in your answer.

Context excerpts:
{context}

Question: {query}

Answer:
    """

    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

    t_llm_start = time.perf_counter()
    response = llm.invoke(prompt)
    llm_ms = (time.perf_counter() - t_llm_start) * 1000

    sources = []
    for i, doc_and_score in enumerate(docs_and_scores, start=1):
        doc, score = doc_and_score
        page_num = display_page(doc)
        sources.append(
            {
                "index": i,
                "page": page_num,
                "preview": doc.page_content[:300],
                "score": float(score),
            }
        )

    retrieval = {
        "min_score": float(min_score) if min_score is not None else None,
        "avg_score": float(avg_score) if avg_score is not None else None,
        "spread_ratio": float(spread_ratio) if spread_ratio is not None else None,
        "weak_retrieval": weak_retrieval,
        "spread_threshold": spread_threshold,
        "weak_min_avg_score": min_avg,
        "weak_min_best_score": min_best,
        "context_chars": total_chars,
        "retrieval_ms": round(retrieval_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        "max_distance": max_distance,
        "max_distance_triggered": max_distance_triggered,
    }

    return {"answer": response.content, "sources": sources, "retrieval": retrieval}
