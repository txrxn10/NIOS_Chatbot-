import json
import re
import hashlib
from typing import Tuple, List, Dict, Any, Generator, Optional
import boto3
import psycopg2
import redis
from src.config import (
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    EMBEDDING_MODEL_ID,
    GENERATION_MODEL_ID,
    DB_CONFIG,
)

# ---------------------------------------------------------------------------
# Redis Client & Cache Configuration
# ---------------------------------------------------------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
CACHE_TTL_CHUNKS = 3600       # 1 hour TTL for retrieved chunks
CACHE_TTL_ANSWER = 7200       # 2 hours TTL for synthesized answers

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=1,
    )
except Exception:
    redis_client = None

def _get_query_cache_key(prefix: str, query: str, top_k: int = 10) -> str:
    """Generates a stable cache key using normalized text and SHA-256."""
    normalized = " ".join(query.strip().lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"nios:{prefix}:{digest}:k{top_k}"

# ---------------------------------------------------------------------------
# AWS Bedrock Client
# ---------------------------------------------------------------------------
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def get_titan_query_embedding(query_text: str) -> list[float]:
    payload = {
        "inputText": query_text[:8000],
        "dimensions": 1024,
        "normalize": True
    }
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    
    return json.loads(response["body"].read())["embedding"]

# ---------------------------------------------------------------------------
# Core Hybrid RRF Search (Direct Database Execution)
# ---------------------------------------------------------------------------
def _execute_hybrid_rrf_db(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Direct PostgreSQL vector + BM25 hybrid search execution."""
    query_vector = get_titan_query_embedding(query_text)
    
    # Clean query string for full text parsing
    cleaned_query = re.sub(r'[^\w\s\/\-]', ' ', query_text).strip()

    sql = """
    WITH dense_search AS (
        SELECT id, chunk_id, archetype, document, category, topic, content, 
               COALESCE(context_prepended_text, content) AS context,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS dense_rank
        FROM chatbot_knowledge_chunks
        LIMIT 60
    ),
    sparse_search AS (
        SELECT id, chunk_id, archetype, document, category, topic, content, 
               COALESCE(context_prepended_text, content) AS context,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv_content, websearch_to_tsquery('english', %s)) DESC) AS sparse_rank
        FROM chatbot_knowledge_chunks
        WHERE tsv_content @@ websearch_to_tsquery('english', %s)
           OR tsv_content @@ plainto_tsquery('english', %s)
        LIMIT 60
    )
    SELECT 
        COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
        COALESCE(d.archetype, s.archetype) AS archetype,
        COALESCE(d.document, s.document) AS document,
        COALESCE(d.category, s.category) AS category,
        COALESCE(d.topic, s.topic) AS topic,
        COALESCE(d.content, s.content) AS content,
        COALESCE(d.context, s.context) AS context,
        (COALESCE(1.0 / (60 + d.dense_rank), 0.0) + COALESCE(1.0 / (60 + s.sparse_rank), 0.0)) AS rrf_score,
        d.dense_rank,
        s.sparse_rank
    FROM dense_search d
    FULL OUTER JOIN sparse_search s ON d.id = s.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET hnsw.ef_search = 40;")
    cur.execute(sql, (query_vector, cleaned_query, cleaned_query, cleaned_query, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "chunk_id": r[0],
            "archetype": r[1],
            "document": r[2],
            "category": r[3],
            "topic": r[4],
            "content": r[5],
            "context": r[6],
            "rrf_score": float(r[7]),
            "dense_rank": r[8],
            "sparse_rank": r[9]
        }
        for r in rows
    ]

# ---------------------------------------------------------------------------
# Cached Hybrid Search Wrapper
# ---------------------------------------------------------------------------
def hybrid_rrf_search(query_text: str, top_k: int = 10, return_cache_meta: bool = False):
    """
    Retrieves context chunks with Redis tier-4 caching.
    Returns:
        - If return_cache_meta=False (default): List[Dict[str, Any]]
        - If return_cache_meta=True: (List[Dict[str, Any]], bool) indicating (chunks, is_cache_hit)
    """
    cache_key = _get_query_cache_key("chunks", query_text, top_k)
    
    # 1. Check Redis Cache
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                chunks = json.loads(cached_data)
                return (chunks, True) if return_cache_meta else chunks
        except redis.RedisError:
            pass  # Fall through to DB on cache exception

    # 2. Cache Miss: Execute fresh search on PostgreSQL
    chunks = _execute_hybrid_rrf_db(query_text, top_k)

    # 3. Store in Redis
    if redis_client and chunks:
        try:
            redis_client.setex(cache_key, CACHE_TTL_CHUNKS, json.dumps(chunks))
        except redis.RedisError:
            pass

    return (chunks, False) if return_cache_meta else chunks

# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are the official NIOS AI Assistant. "
    "Answer the student's question factually and concisely using ONLY the provided official context.\n"
    "STRICT GROUNDING RULES:\n"
    "1. State exact names, dates, subject codes, fees, and directives explicitly.\n"
    "2. Distinguish clearly between physical premises/addresses and financial institutions.\n"
    "3. CRITICAL: Never mistake Theory examinations for Practical examinations. If a date sheet or document specifies "
    "'Theory Examination', you must NEVER present those dates as practical examination dates. If the user asks about practical "
    "examinations and the context only contains theory examination dates, you must explicitly state that the dates provided are for "
    "the Theory examination and that practical examination schedules or rules are not available in the provided document.\n"
    "4. Do not speculate, extrapolate, or confirm false premises in the student query."
)

def synthesize_answer(query: str, retrieved_chunks: list) -> str:
    """Synthesizes grounded response using Amazon Nova via Bedrock Converse with answer caching."""
    answer_cache_key = _get_query_cache_key("answer", query, top_k=len(retrieved_chunks))

    # Check cache for pre-computed answers on frequent exact queries
    if redis_client:
        try:
            cached_ans = redis_client.get(answer_cache_key)
            if cached_ans:
                return cached_ans
        except redis.RedisError:
            pass

    context_blocks = [
        f"Source Document: {c.get('document', 'NIOS Document')} | Section: {c.get('category', 'General')}\n{c['context']}"
        for c in retrieved_chunks
    ]
    context_str = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Official Context:\n{context_str}\n\nStudent Query: {query}"

    response = bedrock.converse(
        modelId=GENERATION_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": SYSTEM_INSTRUCTION}],
        inferenceConfig={"maxTokens": 600, "temperature": 0.0},
    )

    answer_text = response["output"]["message"]["content"][0]["text"]

    # Cache answer
    if redis_client and answer_text:
        try:
            redis_client.setex(answer_cache_key, CACHE_TTL_ANSWER, answer_text)
        except redis.RedisError:
            pass

    return answer_text


def synthesize_answer_stream(query: str, retrieved_chunks: list) -> Generator[str, None, None]:
    """Streams synthesized response token-by-token using Amazon Nova via Bedrock converse_stream."""
    context_blocks = [
        f"Source Document: {c.get('document', 'NIOS Document')} | Section: {c.get('category', 'General')}\n{c['context']}"
        for c in retrieved_chunks
    ]
    context_str = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Official Context:\n{context_str}\n\nStudent Query: {query}"

    response = bedrock.converse_stream(
        modelId=GENERATION_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": SYSTEM_INSTRUCTION}],
        inferenceConfig={"maxTokens": 1000, "temperature": 0.0},
    )

    for event in response.get("stream", []):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                yield delta["text"]




def synthesize_personalized_answer(
    query: str,
    retrieved_chunks: list,
    student_profile: Dict[str, Any]
) -> str:
    """
    Fuses student-specific factual state with official public circular chunks.
    """
    # 1. Format the verified personal context block
    student_block = (
        f"VERIFIED LOGGED-IN STUDENT DATA:\n"
        f"- Enrollment No: {student_profile.get('enrollment_no')}\n"
        f"- Name: {student_profile.get('student_name')}\n"
        f"- Course Level: {student_profile.get('course_level')}\n"
        f"- Study Centre: {student_profile.get('study_centre_code')}\n"
        f"- Fee Status: {student_profile.get('fee_status')}\n"
        f"- Registered Subjects: {student_profile.get('registered_subjects')}\n"
        f"- Hall Ticket Status: {student_profile.get('hall_ticket_status')}\n"
    )

    # 2. Format public circular context
    context_blocks = [
        f"Source Document: {c.get('document', 'NIOS Document')} | Section: {c.get('category', 'General')}\n{c['context']}"
        for c in retrieved_chunks
    ]
    public_context_str = "\n\n---\n\n".join(context_blocks)

    # 3. Construct personalized prompt
    user_prompt = (
        f"{student_block}\n\n"
        f"OFFICIAL PUBLIC REGULATIONS & DATESHEETS:\n"
        f"{public_context_str}\n\n"
        f"Student Query: {query}\n\n"
        f"Instructions: Answer the student's question by applying the official regulations to their specific registered subjects and enrollment details. "
        f"Address the student directly by name where helpful."
    )

    response = bedrock.converse(
        modelId=GENERATION_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": SYSTEM_INSTRUCTION}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.0},
    )

    return response["output"]["message"]["content"][0]["text"]