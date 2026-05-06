import sqlite3
import os
from config import BASE_OUTPUT

db_path = os.path.join(BASE_OUTPUT, "episodes.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS episodes (
    uid TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    file TEXT,
    downloaded INTEGER DEFAULT 0,
    transcribed INTEGER DEFAULT 0
)
""")

conn.commit()


def add_episode(uid, title, url, file):
    c.execute("""
    INSERT OR IGNORE INTO episodes (uid, title, url, file)
    VALUES (?, ?, ?, ?)
    """, (uid, title, url, file))
    conn.commit()


def get_pending_downloads():
    return c.execute("""
    SELECT uid, url, file FROM episodes WHERE downloaded=0
    """).fetchall()


def get_pending_transcriptions():
    return c.execute("""
    SELECT uid, file FROM episodes
    WHERE downloaded=1 AND transcribed=0
    """).fetchall()


def mark_downloaded(uid):
    c.execute("UPDATE episodes SET downloaded=1 WHERE uid=?", (uid,))
    conn.commit()


def mark_transcribed(uid):
    c.execute("UPDATE episodes SET transcribed=1 WHERE uid=?", (uid,))
    conn.commit()