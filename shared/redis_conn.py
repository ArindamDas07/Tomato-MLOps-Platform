import os
import redis
import sys
from loguru import logger

def get_redis_client() -> redis.Redis:
    """
    Creates and validates a robust Redis connection pool.
    Optimized for high-concurrency MLOps workloads and Kubernetes scaling.
    """
    try:
        # Load config from Environment (Injected by K8s ConfigMap)
        host = os.getenv("REDIS_HOST", "redis")
        port = int(os.getenv("REDIS_PORT", 6379))
        db = int(os.getenv("REDIS_DB_APP", 2))

        # 1. Initialize High-Concurrency Connection Pool
        # We increase max_connections to 50 to support multiple API replicas and Workers
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            max_connections=50, 
            # socket_timeout: prevents the process from hanging if Redis is overloaded
            socket_timeout=5.0,
            # health_check_interval: automatically detects and drops dead connections
            health_check_interval=30 
        )

        client = redis.Redis(connection_pool=pool)

        # 2. Connection Health Check (The "Senior" Ping)
        # Ensuring the circuit is closed before the app starts accepting traffic
        client.ping()
        logger.info(f"✅ Shared Redis Link Established: {host}:{port}/db-{db}")
        return client

    except redis.ConnectionError as e:
        logger.error(f"❌ FATAL: Redis Connection Failure: {e}")
        # Crash immediately so K8s can restart the pod (Fail-fast principle)
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected Error in Redis Client: {e}")
        sys.exit(1)

# Initialize the global client used by both API and Worker
redis_client = get_redis_client()

# Result TTL (Time To Live): Defaulting to 1 hour (3600s)
# This is crucial for preventing memory leaks in Redis.
RESULT_TTL = int(os.getenv("RESULT_TTL", 3600))