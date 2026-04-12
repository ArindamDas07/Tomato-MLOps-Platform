import time
import shutil
import re
from pathlib import Path
from loguru import logger

CLEANUP_TARGET = Path("/app/uploads")
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MAX_AGE_SECONDS = 3600 * 2 

def clean_stale_folders():
    now = time.time()
    logger.info("🧹 Janitor: Starting hygiene check...")
    
    if not CLEANUP_TARGET.exists():
        logger.warning(f"⚠️ Janitor: Target path {CLEANUP_TARGET} is missing.")
        return

    purged_count = 0
    skipped_count = 0

    for user_folder in CLEANUP_TARGET.iterdir():
        try:
            if not user_folder.is_dir():
                continue

            if not UUID_PATTERN.match(user_folder.name):
                logger.debug(f"⏭️ Janitor: Skipping non-session folder: {user_folder.name}")
                skipped_count += 1
                continue

            try:
                folder_age = now - user_folder.stat().st_mtime
                if folder_age > MAX_AGE_SECONDS:
                    shutil.rmtree(user_folder)
                    logger.info(f"♻️ Janitor: Purged abandoned session: {user_folder.name}")
                    purged_count += 1
            except FileNotFoundError:
                continue

        except Exception as e:
            logger.error(f"❌ Janitor error processing {user_folder.name}: {e}")
            continue
    
    if purged_count > 0:
        logger.success(f"🧹 Janitor: Finished. Purged {purged_count} folders.")
    else:
        logger.info("🧹 Janitor: Hygiene check finished. Volume is clean.")

if __name__ == "__main__":
    clean_stale_folders()