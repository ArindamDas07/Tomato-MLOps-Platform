import os
import redis
import sys
from loguru import logger

def get_redis_client() -> redis.Redis:
    """
    Creates and validates a Redis connection pool.
    Using a function-based approach allows for better testing and error handling.
    """
    try:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        db = int(os.getenv("REDIS_DB_APP", 2))

        # 1. Initialize Connection Pool
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            max_connections=20,
            # Senior Move: socket_timeout prevents the API from hanging forever 
            # if the network is flaky
            socket_timeout=5.0 
        )

        client = redis.Redis(connection_pool=pool)

        # 2. Connection Health Check (The "Senior" Ping)
        # We try to ping Redis. If it fails, we catch it now, not during a user request.
        client.ping()
        logger.info(f"Successfully connected to Redis at {host}:{port}/db {db}")
        return client

    except redis.ConnectionError as e:
        logger.error(f"FATAL: Could not connect to Redis: {e}")
        # In a production K8s environment, we exit. 
        # K8s will see the 'CrashLoopBackOff' and try to restart us.
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error initializing Redis: {e}")
        sys.exit(1)

# Initialize the global client
redis_client = get_redis_client()

# Senior Tip: We will use this TTL value in our main.py later
RESULT_TTL = int(os.getenv("RESULT_TTL", 3600))