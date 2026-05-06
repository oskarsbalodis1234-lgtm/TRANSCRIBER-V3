import os


BASE_OUTPUT = os.getenv("DATA_DIR", "data")
MP3_DIR = os.path.join(BASE_OUTPUT, "mp3")
TXT_DIR = os.path.join(BASE_OUTPUT, "txt")
METADATA_FILE = os.path.join(BASE_OUTPUT, "episodes_metadata.json")
ZIP_PATH = os.path.join(BASE_OUTPUT, "anchor_podcast_archive.zip")


def ensure_data_dirs():
    os.makedirs(MP3_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)
