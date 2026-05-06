import os
import zipfile

from config import BASE_OUTPUT, METADATA_FILE, TXT_DIR, ZIP_PATH, ensure_data_dirs


def log_message(message, log=None):
    print(message, flush=True)
    if log:
        log(message)


def zip_and_cleanup(log=None):
    files_added = []
    ensure_data_dirs()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        if os.path.exists(METADATA_FILE):
            try:
                archive.write(METADATA_FILE, os.path.relpath(METADATA_FILE, BASE_OUTPUT))
                files_added.append(os.path.relpath(METADATA_FILE, BASE_OUTPUT))
            except Exception as e:
                log_message(f"Failed adding episodes_metadata.json: {str(e)}", log)

        if os.path.exists(TXT_DIR):
            for root, _, files in os.walk(TXT_DIR):
                for file in files:
                    if not file.endswith(".txt"):
                        continue

                    path = os.path.join(root, file)
                    arcname = os.path.relpath(path, BASE_OUTPUT)
                    try:
                        archive.write(path, arcname)
                        files_added.append(arcname)
                    except Exception as e:
                        log_message(f"Failed adding {file}: {str(e)}", log)

    log_message(f"ZIP created with {len(files_added)} files", log)

    try:
        size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
        log_message(f"ZIP size: {size_mb:.2f} MB", log)
    except Exception as e:
        log_message(f"Could not read ZIP size: {str(e)}", log)

    if log:
        log("ZIP contents (first 20 files):")
        for file in files_added[:20]:
            log(f" - {file}")
