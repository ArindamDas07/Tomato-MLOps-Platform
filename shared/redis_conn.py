import os
import redis
from loguru import logger

def get_redis_client() -> redis.Redis:
    """
    Initializes a Redis connection pool.
    Senior Move: We do NOT ping here. This allows the module to be 
    imported during testing without a live database.
    """
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB_APP", 2))

    pool = redis.ConnectionPool(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        max_connections=50,
        socket_timeout=5.0
    )
    return redis.Redis(connection_pool=pool)

# Global client instance
redis_client = get_redis_client()

def check_redis_health():
    """
    Explicit health check to be called during app startup.
    """
    try:
        redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis Health Check Failed: {e}")
        return False

RESULT_TTL = int(os.getenv("RESULT_TTL", 3600))