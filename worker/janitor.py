import os
import time
import shutil
import re
from pathlib import Path
from loguru import logger

# In K8s, /app/uploads IS the 'active-sessions' folder because of subPath
CLEANUP_TARGET = Path("/app/uploads")
# 2 Hours is a safe bet for MLOps. Most tasks finish in < 2 seconds.
MAX_AGE_SECONDS = 3600 * 2 

# Senior Move: Use Regex to ensure we only delete UUID folders
# This prevents the janitor from accidentally deleting system files
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

def clean_stale_folders():
    now = time.time()
    logger.info("🧹 Janitor: Starting hygiene check...")
    
    if not CLEANUP_TARGET.exists():
        logger.warning(f"⚠️ Janitor: Target path {CLEANUP_TARGET} does not exist.")
        return

    count = 0
    # Use .iterdir() for memory efficiency
    for user_folder in CLEANUP_TARGET.iterdir():
        # 1. Safety Check: Is it a directory?
        if not user_folder.is_dir():
            continue

        # 2. Safety Check: Does the folder name look like a User UUID?
        if not UUID_PATTERN.match(user_folder.name):
            logger.debug(f"⏭️ Janitor: Skipping non-session folder: {user_folder.name}")
            continue

        # 3. Calculate age based on Last Modified Time
        try:
            folder_age = now - user_folder.stat().st_mtime
            
            if folder_age > MAX_AGE_SECONDS:
                # Senior Tip: Log exactly what is being deleted for audit trails
                shutil.rmtree(user_folder)
                logger.info(f"♻️ Janitor: Purged stale session: {user_folder.name} (Age: {int(folder_age/60)} mins)")
                count += 1
        except Exception as e:
            logger.error(f"❌ Janitor: Failed to delete {user_folder.name}: {e}")
    
    if count > 0:
        logger.success(f"🧹 Janitor: Hygiene check finished. Purged {count} folders.")
    else:
        logger.info("🧹 Janitor: Hygiene check finished. Everything is clean.")

if __name__ == "__main__":
    clean_stale_folders()