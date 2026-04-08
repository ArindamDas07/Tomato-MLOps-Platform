# app/redis_conn.py
import os
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB_APP = int(os.getenv("REDIS_DB_APP", 2))

# Create a connection pool
pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB_APP,
    decode_responses=True,
    max_connections=20  # Limit connections to prevent overwhelming Redis
)

# Use the pool
redis_client = redis.Redis(connection_pool=pool)