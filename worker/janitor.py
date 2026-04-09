import os
import time
import shutil
from pathlib import Path
from loguru import logger

# This path must match the volume mount in K8s
UPLOAD_DIR = Path("/app/uploads")
MAX_AGE_SECONDS = 3600 * 2  # 2 Hours

def clean_stale_folders():
    now = time.time()
    logger.info("🧹 Janitor: Starting storage hygiene check...")
    
    if not UPLOAD_DIR.exists():
        logger.warning("Janitor: Upload directory not found.")
        return

    count = 0
    for folder in UPLOAD_DIR.iterdir():
        if folder.is_dir():
            # Calculate folder age
            folder_age = now - folder.stat().st_mtime
            
            if folder_age > MAX_AGE_SECONDS:
                try:
                    shutil.rmtree(folder)
                    logger.info(f"♻️ Janitor: Purged stale session: {folder.name}")
                    count += 1
                except Exception as e:
                    logger.error(f"Janitor: Failed to delete {folder.name}: {e}")
    
    logger.success(f"🧹 Janitor: Cleanup finished. Removed {count} folders.")

if __name__ == "__main__":
    clean_stale_folders()