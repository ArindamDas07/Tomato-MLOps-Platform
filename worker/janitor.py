import os
import time
import shutil
import re
from pathlib import Path
from loguru import logger

# In K8s, /app/uploads is the shared volume mount point
CLEANUP_TARGET = Path("/app/uploads")

# Senior Move: Ensuring we only touch folders created by our UUID logic.
# This prevents accidental deletion of system files or hidden mount metadata.
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# We consider a session "abandoned" if no activity for 2 hours.
MAX_AGE_SECONDS = 3600 * 2 

def clean_stale_folders():
    """
    Scans the shared volume and removes session folders that were not 
    cleaned up by the API (e.g., due to browser shutdown or network loss).
    """
    now = time.time()
    logger.info("🧹 Janitor: Starting hygiene check...")
    
    if not CLEANUP_TARGET.exists():
        logger.warning(f"⚠️ Janitor: Target path {CLEANUP_TARGET} is missing. Check K8s VolumeMounts.")
        return

    purged_count = 0
    skipped_count = 0

    # Using .iterdir() is more memory-efficient than os.listdir() for large volumes
    for user_folder in CLEANUP_TARGET.iterdir():
        try:
            # 1. Safety Check: Is it a directory and does it match our naming convention?
            if not user_folder.is_dir():
                continue

            if not UUID_PATTERN.match(user_folder.name):
                logger.debug(f"⏭️ Janitor: Skipping non-session folder: {user_folder.name}")
                skipped_count += 1
                continue

            # 2. Age Check: How long since the folder was last modified?
            # We wrap this in a sub-try because the folder could be deleted 
            # by the API /result call exactly now (Race Condition).
            try:
                folder_age = now - user_folder.stat().st_mtime
                
                if folder_age > MAX_AGE_SECONDS:
                    shutil.rmtree(user_folder)
                    logger.info(f"♻️ Janitor: Purged abandoned session: {user_folder.name} (Age: {int(folder_age/60)} mins)")
                    purged_count += 1
            except FileNotFoundError:
                # API deleted it already, this is a 'good' race condition.
                continue

        except Exception as e:
            logger.error(f"❌ Janitor encountered an error processing {user_folder.name}: {e}")
            continue
    
    if purged_count > 0:
        logger.success(f"🧹 Janitor: Finished. Purged {purged_count} folders. (Ignored {skipped_count} system folders)")
    else:
        logger.info("🧹 Janitor: Hygiene check finished. Volume is clean.")

if __name__ == "__main__":
    clean_stale_folders()