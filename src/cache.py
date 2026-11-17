import os
import json
import hashlib
from typing import Optional, List, Dict, Any
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DEFAULT_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

def _normalize_query(query: str) -> str:
    """Normalizes query text to ensure minor whitespace/casing variations hit cache."""
    return " ".join(query.strip().lower().split())

def _generate_cache_key(prefix: str, text: str) -> str:
    norm = _normalize_query(text)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"

def get_cached_search_results(query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
    """Retrieves cached RRF search results from Redis."""
    key = f"{_generate_cache_key('faq_search', query)}:k{top_k}"
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except redis.RedisError:
        pass
    return None

def set_cached_search_results(query: str, top_k: int, results: List[Dict[str, Any]], ttl: int = DEFAULT_TTL_SECONDS):
    """Caches RRF search results in Redis."""
    key = f"{_generate_cache_key('faq_search', query)}:k{top_k}"
    try:
        redis_client.setex(key, ttl, json.dumps(results))
    except redis.RedisError:
        pass

def check_rate_limit(client_id: str, limit: int = 60, window: int = 60) -> bool:
    """
    Sliding window rate-limiter: Allows up to `limit` queries per `window` seconds.
    Returns True if allowed, False if exceeded.
    """
    key = f"ratelimit:{client_id}"
    try:
        current = redis_client.incr(key)
        if current == 1:
            redis_client.expire(key, window)
        return current <= limit
    except redis.RedisError:
        return True  # Fail open if cache is unavailable







    