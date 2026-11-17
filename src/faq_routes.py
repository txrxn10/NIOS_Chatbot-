import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

from src.config import DB_CONFIG

router = APIRouter(prefix="/api/v1/faq", tags=["Phase 1 Guided FAQ Navigation"])

# Redis connection for caching taxonomy responses
try:
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=1
    )
except Exception:
    redis_client = None

TAXONOMY_CACHE_TTL = 3600  # 1 hour cache

def _get_db():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

@router.get("/categories", status_code=status.HTTP_200_OK)
def get_categories(lang: str = Query(default="en", pattern="^(en|hi)$")):
    """
    Returns the 7 primary categories with topic counts.
    Cached in Redis to protect origin during peak browsing.
    """
    cache_key = f"nios:taxonomy:categories:{lang}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except redis.RedisError:
            pass

    sql = """
    SELECT 
        c.id, 
        c.display_order, 
        CASE WHEN %s = 'hi' THEN c.name_hi ELSE c.name_en END AS name,
        c.icon_name,
        COUNT(t.id)::int AS topic_count
    FROM chatbot_categories c
    LEFT JOIN chatbot_topics t ON c.id = t.category_id AND t.is_active = TRUE
    WHERE c.is_active = TRUE
    GROUP BY c.id, c.display_order, c.name_en, c.name_hi, c.icon_name
    ORDER BY c.display_order ASC;
    """
    
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (lang,))
            rows = cur.fetchall()
            categories = [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error retrieving categories: {str(e)}"
        )
    finally:
        conn.close()

    result = {
        "status": "success",
        "language": lang,
        "data": categories
    }

    if redis_client:
        try:
            redis_client.setex(cache_key, TAXONOMY_CACHE_TTL, json.dumps(result))
        except redis.RedisError:
            pass

    return result

@router.get("/topics", status_code=status.HTTP_200_OK)
def get_topics(
    category_id: str = Query(..., examples=["admission"]),
    lang: str = Query(default="en", pattern="^(en|hi)$")
):
    """
    Returns topics under a selected category with FAQ count.
    """
    cache_key = f"nios:taxonomy:topics:{category_id}:{lang}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except redis.RedisError:
            pass

    sql = """
    SELECT 
        t.id, 
        CASE WHEN %s = 'hi' THEN t.title_hi ELSE t.title_en END AS title,
        COUNT(f.id)::int AS faq_count
    FROM chatbot_topics t
    LEFT JOIN chatbot_faqs f ON t.id = f.topic_id AND f.is_active = TRUE
    WHERE t.category_id = %s AND t.is_active = TRUE
    GROUP BY t.id, t.title_en, t.title_hi, t.display_order
    ORDER BY t.display_order ASC;
    """
    
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (lang, category_id))
            rows = cur.fetchall()
            topics = [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error retrieving topics: {str(e)}"
        )
    finally:
        conn.close()

    result = {
        "status": "success",
        "category_id": category_id,
        "language": lang,
        "topics": topics
    }

    if redis_client:
        try:
            redis_client.setex(cache_key, TAXONOMY_CACHE_TTL, json.dumps(result))
        except redis.RedisError:
            pass

    return result

@router.get("/detail", status_code=status.HTTP_200_OK)
def get_topic_faqs(
    topic_id: str = Query(..., examples=["adm_eligibility"]),
    lang: str = Query(default="en", pattern="^(en|hi)$")
):
    """
    Returns curated FAQ Q&A accordions with official source links.
    """
    cache_key = f"nios:taxonomy:faqs:{topic_id}:{lang}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except redis.RedisError:
            pass

    sql = """
    SELECT 
        id::text,
        CASE WHEN %s = 'hi' THEN question_hi ELSE question_en END AS question,
        CASE WHEN %s = 'hi' THEN answer_hi ELSE answer_en END AS answer,
        CASE WHEN %s = 'hi' THEN source_url_hi ELSE source_url_en END AS source_url,
        related_faq_ids
    FROM chatbot_faqs
    WHERE topic_id = %s AND is_active = TRUE
    ORDER BY view_count DESC
    LIMIT 4;
    """
    
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (lang, lang, lang, topic_id))
            rows = cur.fetchall()
            faqs = [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error retrieving FAQ detail: {str(e)}"
        )
    finally:
        conn.close()

    result = {
        "status": "success",
        "topic_id": topic_id,
        "language": lang,
        "faqs": faqs
    }

    if redis_client:
        try:
            redis_client.setex(cache_key, TAXONOMY_CACHE_TTL, json.dumps(result))
        except redis.RedisError:
            pass

    return result