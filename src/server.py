import time
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.retriever import (
    hybrid_rrf_search,
    synthesize_answer,
    synthesize_answer_stream
)
from src.limiter import check_sliding_window_rate_limit
from src.faq_routes import router as faq_router
from src.auth import get_current_student
from src.student_service import get_student_context_by_enrollment
from src.templates import (
    format_admission_status,
    format_exam_schedule,
    format_results,
    format_payment_history,
    format_grievances
)

app = FastAPI(
    title="NIOS Conversational Chatbot API",
    version="1.0.0",
    description="High-concurrency Phase 1 Guided FAQ & Grounded RAG Backend"
)

# ---------------------------------------------------------------------------
# Cross-Origin Resource Sharing (CORS) Configuration
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate-Limiting & Telemetry Middleware (Redis Sliding-Window)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_and_telemetry_middleware(request: Request, call_next):
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

    allowed, remaining, retry_after = check_sliding_window_rate_limit(
        identifier=client_ip,
        max_requests=60,
        window_seconds=60
    )

    if not allowed:
        return Response(
            content='{"status": "error", "message": "Rate limit exceeded. Please try again shortly."}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": "60",
                "X-RateLimit-Remaining": "0",
            }
        )

    t0 = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - t0) * 1000

    response.headers["X-RateLimit-Limit"] = "60"
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response

# ---------------------------------------------------------------------------
# Mount Sub-Routers
# ---------------------------------------------------------------------------
app.include_router(faq_router)

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is the fee for Senior Secondary theory examination?"])
    top_k: int = Field(default=10, ge=1, le=20)

class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    cache_hit: bool
    retrieval_latency_ms: float
    sources: List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# Core Service Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "service": "nios-rag-backend"}

@app.post("/api/v1/chat/query", response_model=QueryResponse)
def handle_chat_query(
    payload: QueryRequest,
    student_auth: Optional[Dict[str, Any]] = Depends(get_current_student)
):
    """
    Unified Ingress Gateway:
      - Authenticated personal requests -> Modules 1 to 5 Deterministic Templates
      - Unauthenticated personal requests -> Sign-in prompt
      - Public questions -> Hybrid RRF Vector Search & Bedrock Synthesis
    """
    t0 = time.time()
    query_lower = payload.query.lower()

    # Detect Phase 2 Module 1-5 Intent Triggers
    personal_keywords = [
        "my exam", "my subjects", "when is my", "hall ticket",
        "my admission", "my status", "admission status", "my stream",
        "my result", "my marks", "marksheet", "did i pass",
        "my fee", "fee receipt", "payment history", "my payment",
        "my grievance", "my complaint", "ticket status", "e-service"
    ]
    is_personal_query = any(k in query_lower for k in personal_keywords)

    if is_personal_query:
        if not student_auth:
            return QueryResponse(
                status="success",
                query=payload.query,
                answer="To view your personal details, examination dates, results, fee ledger, or ticket status, please sign in with your student credentials.",
                cache_hit=False,
                retrieval_latency_ms=round((time.time() - t0) * 1000, 2),
                sources=[]
            )

        student_data = get_student_context_by_enrollment(student_auth["enrollment_no"])
        if not student_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student record for enrollment {student_auth['enrollment_no']} not found."
            )

        # Route by specific module intent
        if any(k in query_lower for k in ["result", "marks", "marksheet", "pass"]):
            answer = format_results(student_data)
        elif any(k in query_lower for k in ["fee", "payment", "paid", "receipt", "dues"]):
            answer = format_payment_history(student_data)
        elif any(k in query_lower for k in ["grievance", "complaint", "ticket", "e-service"]):
            answer = format_grievances(student_data)
        elif any(k in query_lower for k in ["admission", "status", "stream", "document"]):
            answer = format_admission_status(student_data)
        else:
            answer = format_exam_schedule(student_data)

        return QueryResponse(
            status="success",
            query=payload.query,
            answer=answer,
            cache_hit=False,
            retrieval_latency_ms=round((time.time() - t0) * 1000, 2),
            sources=[{
                "chunk_id": "SDMIS_DIRECT_RECORD",
                "document": "SDMIS Student Records",
                "category": "Authenticated Student Context",
                "rrf_score": 1.0
            }]
        )

    # Phase 1 Public Hybrid Search Path
    chunks, cache_hit = hybrid_rrf_search(
        payload.query,
        top_k=payload.top_k,
        return_cache_meta=True
    )
    retrieval_ms = (time.time() - t0) * 1000

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching NIOS official records found."
        )

    answer = synthesize_answer(payload.query, chunks)

    sources = [
        {
            "chunk_id": c.get("chunk_id"),
            "document": c.get("document"),
            "category": c.get("category"),
            "rrf_score": round(c.get("rrf_score", 0.0), 4),
        }
        for c in chunks[:4]
    ]

    return QueryResponse(
        status="success",
        query=payload.query,
        answer=answer,
        cache_hit=cache_hit,
        retrieval_latency_ms=round(retrieval_ms, 2),
        sources=sources
    )

@app.post("/api/v1/chat/query/stream")
def handle_chat_query_stream(
    payload: QueryRequest,
    student_auth: Optional[Dict[str, Any]] = Depends(get_current_student)
):
    """
    Server-Sent Events (SSE) streaming endpoint for live token rendering in UI widgets.
    """
    t0 = time.time()
    query_lower = payload.query.lower()

    personal_keywords = [
        "my exam", "my subjects", "when is my", "hall ticket",
        "my admission", "my status", "admission status", "my stream",
        "my result", "my marks", "marksheet", "did i pass",
        "my fee", "fee receipt", "payment history", "my payment",
        "my grievance", "my complaint", "ticket status", "e-service"
    ]
    is_personal_query = any(k in query_lower for k in personal_keywords)

    if is_personal_query:
        if not student_auth:
            def event_generator_unauth():
                meta_json = json.dumps({"type": "meta", "cache_hit": False, "retrieval_latency_ms": 1.0, "sources": []})
                yield f"data: {meta_json}\n\n"
                msg = "To view your personal details, examination dates, results, fee ledger, or ticket status, please sign in with your student credentials."
                for word in msg.split(" "):
                    token_json = json.dumps({"type": "token", "delta": word + " "})
                    yield f"data: {token_json}\n\n"
                    time.sleep(0.02)
                yield 'data: {"type": "done"}\n\n'
            return StreamingResponse(event_generator_unauth(), media_type="text/event-stream")

        student_data = get_student_context_by_enrollment(student_auth["enrollment_no"])
        if any(k in query_lower for k in ["result", "marks", "marksheet", "pass"]):
            answer = format_results(student_data)
        elif any(k in query_lower for k in ["fee", "payment", "paid", "receipt", "dues"]):
            answer = format_payment_history(student_data)
        elif any(k in query_lower for k in ["grievance", "complaint", "ticket", "e-service"]):
            answer = format_grievances(student_data)
        elif any(k in query_lower for k in ["admission", "status", "stream", "document"]):
            answer = format_admission_status(student_data)
        else:
            answer = format_exam_schedule(student_data)

        sources = [{
            "chunk_id": "SDMIS_DIRECT_RECORD",
            "document": "SDMIS Student Records",
            "category": "Authenticated Student Context",
            "rrf_score": 1.0
        }]

        def event_generator_personal():
            meta_json = json.dumps({"type": "meta", "cache_hit": False, "retrieval_latency_ms": 1.5, "sources": sources})
            yield f"data: {meta_json}\n\n"
            for line in answer.split("\n"):
                token_json = json.dumps({"type": "token", "delta": line + "\n"})
                yield f"data: {token_json}\n\n"
                time.sleep(0.01)
            yield 'data: {"type": "done"}\n\n'
        return StreamingResponse(event_generator_personal(), media_type="text/event-stream")

    # Public Hybrid Search Streaming
    chunks, cache_hit = hybrid_rrf_search(
        payload.query,
        top_k=payload.top_k,
        return_cache_meta=True
    )
    retrieval_ms = (time.time() - t0) * 1000

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching NIOS official records found."
        )

    sources = [
        {
            "chunk_id": c.get("chunk_id"),
            "document": c.get("document"),
            "category": c.get("category"),
            "rrf_score": round(c.get("rrf_score", 0.0), 4),
        }
        for c in chunks[:4]
    ]

    def event_generator():
        init_payload = {
            "type": "meta",
            "cache_hit": cache_hit,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "sources": sources
        }
        meta_json = json.dumps(init_payload)
        yield f"data: {meta_json}\n\n"

        for delta in synthesize_answer_stream(payload.query, chunks):
            token_json = json.dumps({"type": "token", "delta": delta})
            yield f"data: {token_json}\n\n"

        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )