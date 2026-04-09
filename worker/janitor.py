import os
import time
import shutil
from pathlib import Path
from loguru import logger

# In K8s, /app/uploads IS the 'active-sessions' folder because of subPath
CLEANUP_TARGET = Path("/app/uploads")
MAX_AGE_SECONDS = 3600 * 2 # 2 Hours

def clean_stale_folders():
    now = time.time()
    logger.info("🧹 Janitor: Starting hygiene check...")
    
    if not CLEANUP_TARGET.exists():
        return

    count = 0
    for user_folder in CLEANUP_TARGET.iterdir():
        if user_folder.is_dir():
            # Calculate age
            folder_age = now - user_folder.stat().st_mtime
            
            if folder_age > MAX_AGE_SECONDS:
                try:
                    shutil.rmtree(user_folder)
                    logger.info(f"♻️ Janitor: Deleted expired session: {user_folder.name}")
                    count += 1
                except Exception as e:
                    logger.error(f"Janitor: Failed to delete {user_folder.name}: {e}")
    
    logger.success(f"🧹 Janitor: Finished. Purged {count} stale user sessions.")

if __name__ == "__main__":
    clean_stale_folders()