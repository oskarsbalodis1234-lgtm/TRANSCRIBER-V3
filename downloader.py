import json
import os
from hashlib import md5

import requests
from bs4 import BeautifulSoup

from config import METADATA_FILE, MP3_DIR, ensure_data_dirs


ensure_data_dirs()

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

episodes = []


def ingest_rss(rss_url):
    episodes.clear()

    response = session.get(rss_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml-xml")

    for i, item in enumerate(soup.find_all("item")):
        title = item.title.text if item.title else f"episode_{i}"
        enclosure = item.find("enclosure")

        if not enclosure:
            continue

        url = enclosure.get("url")
        if not url:
            continue

        uid = md5((title + url).encode()).hexdigest()
        file = f"{uid}.mp3"

        episodes.append(
            {
                "uid": uid,
                "url": url,
                "file": file,
                "title": title,
                "episode_number": len(episodes) + 1,
            }
        )

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)


def run_downloads(log=None):
    total = len(episodes)

    for i, episode in enumerate(episodes, start=1):
        path = os.path.join(MP3_DIR, episode["file"])

        if os.path.exists(path):
            continue

        msg = f"Download {i}/{total}: {episode['title']}"
        print(msg, flush=True)

        if log:
            log(msg)

        with session.get(episode["url"], stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
