import os
import time
import shutil
from pathlib import Path
from loguru import logger

# --- THE FIX: Janitor only scans the "Trash Zone" ---
CLEANUP_TARGET = Path("/app/uploads/active-sessions")
MAX_AGE_SECONDS = 3600 * 2 

def clean_stale_folders():
    now = time.time()
    logger.info(f"🧹 Janitor: Scanning {CLEANUP_TARGET} for stale data...")
    
    if not CLEANUP_TARGET.exists():
        logger.warning("Janitor: Target directory does not exist yet.")
        return

    count = 0
    # Iterates through /app/uploads/active-sessions/{user_id}
    for user_folder in CLEANUP_TARGET.iterdir():
        if user_folder.is_dir():
            folder_age = now - user_folder.stat().st_mtime
            
            if folder_age > MAX_AGE_SECONDS:
                try:
                    shutil.rmtree(user_folder)
                    logger.info(f"♻️ Janitor: Deleted expired session: {user_folder.name}")
                    count += 1
                except Exception as e:
                    logger.error(f"Janitor: Failed to delete {user_folder.name}: {e}")
    
    logger.success(f"🧹 Janitor: Finished. Purged {count} stale user sessions.")