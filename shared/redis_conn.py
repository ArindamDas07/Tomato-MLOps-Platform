import os
import redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import (
    BusyLoadingError, 
    ConnectionError, 
    TimeoutError
)
from loguru import logger

def get_redis_client() -> redis.Redis:
    """
    Initializes a Redis connection pool with built-in resilience.
    
    Senior Move 1: We do NOT ping here. This allows the module to be 
    imported during testing or CI/CD without a live database.
    
    Senior Move 2: Exponential Backoff Retry. Handles transient network
    blips or container restarts automatically.
    """
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB_APP", 2))

    # Define the retry strategy: 3 attempts with exponential wait times
    retry_strategy = Retry(ExponentialBackoff(), 3)

    # Initialize the Connection Pool
    pool = redis.ConnectionPool(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        max_connections=50, # Limits total connections to prevent overhead
        socket_timeout=5.0   # Prevents the app from hanging on a dead link
    )

    # Return the client with the retry logic attached
    return redis.Redis(
        connection_pool=pool,
        retry=retry_strategy,
        retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError]
    )

# Global singleton client instance
redis_client = get_redis_client()

def check_redis_health():
    """
    Explicit health check for Kubernetes Liveness/Readiness probes.
    """
    try:
        redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"❌ Redis Health Check Failed: {e}")
        return False

# Time-to-Live for results: Default 1 hour
RESULT_TTL = int(os.getenv("RESULT_TTL", 3600))