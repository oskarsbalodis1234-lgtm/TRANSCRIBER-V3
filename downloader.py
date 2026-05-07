import json
import os
from hashlib import md5
from config import METADATA_FILE, MP3_DIR, ensure_data_dirs

ensure_data_dirs()

def ingest_rss(rss_url):
    import requests
    from bs4 import BeautifulSoup

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    episodes = []

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
    return episodes

def download_episode(args):
    episode, i, total, session, log, check_cancel = args
    path = os.path.join(MP3_DIR, episode["file"])

    if check_cancel and check_cancel():
        return

    if os.path.exists(path):
        return

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

def run_downloads(episode_list, log=None, check_cancel=None):
    import concurrent.futures
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    total = len(episode_list)
    
    # Download 2 episodes at once to save memory on Koyeb Free Tier
    download_tasks = [(ep, i, total, session, log, check_cancel) for i, ep in enumerate(episode_list, start=1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download_episode, download_tasks))
