import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    EMBEDDING_MODEL_ID,
    DB_CONFIG,
)

INPUT_CHUNKS_FILE = "data/auto_batch_chunks.json"
CACHE_FILE = "data/titan_batch_embeddings_cache.json"
CONCURRENT_WORKERS = 20
BATCH_INSERT_PAGE_SIZE = 1000

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def init_db_schema():
    """Initializes HNSW vector extension, document registry, and chunks table."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Document registry for delta deduplication
    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_registry (
        filename TEXT PRIMARY KEY,
        sha256_hash TEXT NOT NULL,
        chunk_count INT NOT NULL,
        ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """)

    # Primary knowledge table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chatbot_knowledge_chunks (
        id SERIAL PRIMARY KEY,
        chunk_id TEXT UNIQUE NOT NULL,
        archetype TEXT,
        document TEXT,
        category TEXT,
        topic TEXT,
        content TEXT,
        context_prepended_text TEXT,
        embedding vector(1024),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """)

    # HNSW Index for sub-second logarithmic vector search
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_chatbot_hnsw 
    ON chatbot_knowledge_chunks 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
    """)

    conn.commit()
    cur.close()
    conn.close()

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_titan_embedding_with_retry(text: str, max_retries: int = 5) -> list[float]:
    """Invokes Titan Embeddings v2 with exponential backoff for rate-limit resilience."""
    payload = {
        "inputText": text[:8000],
        "dimensions": 1024,
        "normalize": True
    }
    backoff = 0.5
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )
            return json.loads(response["body"].read())["embedding"]
        except Exception as e:
            if "ThrottlingException" in str(e) or "TooManyRequests" in str(e):
                time.sleep(backoff)
                backoff *= 2
            else:
                if attempt == max_retries - 1:
                    print(f"[Embedding Error] Failed after {max_retries} attempts: {e}")
                    raise e
                time.sleep(backoff)
    return []

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

def run_concurrent_embeddings(chunks: list, cache: dict) -> dict:
    """Generates embeddings using a thread pool for maximum throughput."""
    uncached_chunks = []
    for c in chunks:
        c_hash = compute_sha256(c["text"])
        if c_hash not in cache:
            uncached_chunks.append((c_hash, c["text"]))

    if not uncached_chunks:
        print(f"All {len(chunks)} chunks already present in Titan embedding cache.")
        return cache

    print(f"Processing {len(uncached_chunks)} uncached chunks across {CONCURRENT_WORKERS} parallel workers...")
    start_time = time.time()

    completed = 0
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        future_to_hash = {
            executor.submit(get_titan_embedding_with_retry, text): h
            for h, text in uncached_chunks
        }
        for future in as_completed(future_to_hash):
            h = future_to_hash[future]
            try:
                emb = future.result()
                if emb:
                    cache[h] = emb
                    completed += 1
                    if completed % 50 == 0 or completed == len(uncached_chunks):
                        print(f"  Parallel Embeddings: {completed}/{len(uncached_chunks)} completed.")
            except Exception as e:
                print(f"  [Worker Error] Failed chunk hash {h}: {e}")

    elapsed = time.time() - start_time
    print(f"Embedding generation completed in {elapsed:.2f}s ({len(uncached_chunks)/max(elapsed, 0.001):.1f} chunks/sec).")
    save_cache(cache)
    return cache

def bulk_upsert_chunks(chunks: list, cache: dict):
    """Performs bulk upsert into PostgreSQL using execute_values."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    records = []
    for c in chunks:
        meta = c.get("metadata", {})
        c_hash = compute_sha256(c["text"])
        embedding = cache.get(c_hash)
        if not embedding:
            continue

        records.append((
            c["chunk_id"],
            meta.get("type", "prose_section"),
            meta.get("source", "NIOS Document"),
            meta.get("section", "General"),
            meta.get("doc_title", ""),
            c["text"],
            c["text"],
            embedding
        ))

    upsert_query = """
    INSERT INTO chatbot_knowledge_chunks (
        chunk_id, archetype, document, category, topic, content, context_prepended_text, embedding
    )
    VALUES %s
    ON CONFLICT (chunk_id) DO UPDATE SET
        archetype = EXCLUDED.archetype,
        document = EXCLUDED.document,
        category = EXCLUDED.category,
        topic = EXCLUDED.topic,
        content = EXCLUDED.content,
        context_prepended_text = EXCLUDED.context_prepended_text,
        embedding = EXCLUDED.embedding;
    """

    print(f"Writing {len(records)} chunks into pgvector via bulk transaction...")
    start_time = time.time()
    execute_values(cur, upsert_query, records, page_size=BATCH_INSERT_PAGE_SIZE)
    conn.commit()

    # Refresh full-text GIN index column
    print("Rebuilding lexical GIN tsvector...")
    cur.execute("""
    ALTER TABLE chatbot_knowledge_chunks DROP COLUMN IF EXISTS tsv_content;
    ALTER TABLE chatbot_knowledge_chunks 
    ADD COLUMN tsv_content tsvector 
    GENERATED ALWAYS AS (
        to_tsvector('english', 
            coalesce(document, '') || ' ' || 
            coalesce(category, '') || ' ' || 
            coalesce(topic, '') || ' ' || 
            coalesce(context_prepended_text, content, '')
        )
    ) STORED;
    CREATE INDEX IF NOT EXISTS idx_chatbot_tsv ON chatbot_knowledge_chunks USING GIN(tsv_content);
    """)
    conn.commit()

    elapsed = time.time() - start_time
    print(f"Database insertion and GIN indexing completed in {elapsed:.2f}s.")

    cur.close()
    conn.close()

def main():
    if not os.path.exists(INPUT_CHUNKS_FILE):
        print(f"Error: {INPUT_CHUNKS_FILE} not found. Run python src/auto_batch_ingest.py first.")
        return

    init_db_schema()

    with open(INPUT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {INPUT_CHUNKS_FILE}.")
    cache = load_cache()
    cache = run_concurrent_embeddings(chunks, cache)
    bulk_upsert_chunks(chunks, cache)
    print("\nProduction ingestion pipeline execution complete.")

if __name__ == "__main__":
    main()