import time
import os
from typing import Tuple
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=1
)

def check_sliding_window_rate_limit(
    identifier: str,
    max_requests: int = 60,
    window_seconds: int = 60
) -> Tuple[bool, int, int]:
    if not redis_client:
        return True, max_requests, 0

    now = time.time()
    clear_before = now - window_seconds
    key = f"nios:ratelimit:{identifier}"

    try:
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, clear_before)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window_seconds)
        
        _, current_count, _, _ = pipe.execute()

        if current_count < max_requests:
            remaining = max_requests - (current_count + 1)
            return True, max(0, remaining), 0
        else:
            oldest = redis_client.zrange(key, 0, 0, withscores=True)
            retry_after = int(window_seconds - (now - oldest[0][1])) if oldest else window_seconds
            return False, 0, max(1, retry_after)

    except redis.RedisError:
        return True, max_requests, 0