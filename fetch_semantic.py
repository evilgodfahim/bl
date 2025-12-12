#!/usr/bin/env python3
# final_optimized_rss.py
# Python 3.11+ recommended

import feedparser
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import base64

# -------------------- CONFIG --------------------
FEEDS = [
    "http://www.thedailystar.net/latest/rss/rss.xml",
    "https://tbsnews.net/top-news/rss.xml",
    "https://www.dhakatribune.com/feed/"
]

RESULT_FILE = "result.xml"
MAX_ITEMS = 1000          # total items kept
MAX_FEED_ITEMS = 50       # max items checked/added per feed per run
MAX_EXISTING = 50         # compare only against newest this many existing items
SIM_THRESHOLD = 0.88      # cosine similarity threshold for duplicate
DENY_PARTS = ("/sport/", "/sports/", "/entertainment/")

# -------------------- HELPERS --------------------
def b64_encode_array(arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")

def b64_decode_array(s: str) -> np.ndarray:
    b = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(b, dtype=np.float32)

def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

# -------------------- MODEL (load once) --------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    # returns float32 numpy array shape (n, d)
    arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(arr, dtype=np.float32)

# -------------------- XML IO --------------------
def ensure_xml():
    if not os.path.exists(RESULT_FILE):
        root = ET.Element("rss")
        root.set("version", "2.0")
        ch = ET.SubElement(root, "channel")
        # store feed meta container
        ET.SubElement(ch, "feedmeta")
        tree = ET.ElementTree(root)
        tree.write(RESULT_FILE, encoding="utf-8", xml_declaration=True)

def load_tree_and_channel():
    ensure_xml()
    tree = ET.parse(RESULT_FILE)
    root = tree.getroot()
    channel = root.find("channel")
    # ensure feedmeta exists
    if channel.find("feedmeta") is None:
        channel.insert(0, ET.Element("feedmeta"))
    return tree, channel

# -------------------- existing cached embeddings (newest first) --------------------
def get_existing_cached(channel):
    items = channel.findall("item")
    top_items = items[:MAX_EXISTING]
    titles = []
    embeds = []
    for it in top_items:
        t = it.findtext("title", "") or ""
        emb_tag = it.findtext("embed", "")
        if emb_tag:
            try:
                vec = b64_decode_array(emb_tag)
            except Exception:
                vec = embed_texts([t])[0]
        else:
            vec = embed_texts([t])[0]
        titles.append(t)
        embeds.append(vec)
    if embeds:
        emb_mat = np.vstack(embeds).astype(np.float32)
        emb_mat = normalize_matrix(emb_mat)
    else:
        emb_mat = np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return titles, emb_mat

# -------------------- feed meta handling --------------------
def get_feed_lastseen(channel, feed_url):
    fm = channel.find("feedmeta")
    for fm_child in fm.findall("feed"):
        if fm_child.get("url") == feed_url:
            return fm_child.findtext("last_link", "")
    return ""

def set_feed_lastseen(channel, feed_url, last_link):
    fm = channel.find("feedmeta")
    for fm_child in fm.findall("feed"):
        if fm_child.get("url") == feed_url:
            ln = fm_child.find("last_link")
            if ln is None:
                ln = ET.SubElement(fm_child, "last_link")
            ln.text = last_link
            return
    # create
    new = ET.SubElement(fm, "feed")
    new.set("url", feed_url)
    ln = ET.SubElement(new, "last_link")
    ln.text = last_link

# -------------------- make item element with embed --------------------
def build_item(entry, emb_vec: np.ndarray):
    item = ET.Element("item")
    ET.SubElement(item, "title").text = entry.get("title", "")
    ET.SubElement(item, "link").text = entry.get("link", "")
    ET.SubElement(item, "pubDate").text = entry.get("published", datetime.utcnow().isoformat())
    ET.SubElement(item, "guid").text = entry.get("id", entry.get("link", ""))
    ET.SubElement(item, "embed").text = b64_encode_array(emb_vec)
    return item

# -------------------- vectorized duplicate check --------------------
def batch_is_duplicate(candidate_embs: np.ndarray, existing_normed: np.ndarray, threshold=SIM_THRESHOLD):
    # candidate_embs: (m,d) float32
    # existing_normed: (n,d) float32, already normalized
    if existing_normed.size == 0:
        return np.zeros(candidate_embs.shape[0], dtype=bool)
    cand_norm = normalize_matrix(candidate_embs)
    sims = np.matmul(cand_norm, existing_normed.T)  # (m,n)
    maxs = sims.max(axis=1)
    return maxs >= threshold

# -------------------- main processing --------------------
def process_once():
    tree, channel = load_tree_and_channel()
    existing_titles, existing_normed = get_existing_cached(channel)

    # For speed, keep existing_normed and update as we add new items
    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        entries = feed.entries or []
        if not entries:
            # update lastseen to empty if none
            set_feed_lastseen(channel, feed_url, "")
            continue

        last_seen = get_feed_lastseen(channel, feed_url)
        # Build list of new entries (those before last_seen). Feed entries are typically newest-first.
        new_entries = []
        for e in entries:
            link = e.get("link", "")
            if not link:
                continue
            if last_seen and link == last_seen:
                break
            new_entries.append(e)
            if len(new_entries) >= MAX_FEED_ITEMS:
                break

        # If last_seen is empty and new_entries might be older than MAX_FEED_ITEMS,
        # we still only process up to MAX_FEED_ITEMS (handled above).
        if not new_entries:
            # nothing new; update last seen to latest entry link for future runs
            set_feed_lastseen(channel, feed_url, entries[0].get("link", ""))
            continue

        # Compute embeddings for batch of candidate titles
        candidate_titles = [e.get("title", "") for e in new_entries]
        candidate_embs = embed_texts(candidate_titles)  # shape (m,d)

        # Determine duplicates (vectorized)
        dup_mask = batch_is_duplicate(candidate_embs, existing_normed, SIM_THRESHOLD)

        # Iterate through new_entries in order (newest first) and insert non-duplicates at top
        added_any = False
        for idx, entry in enumerate(new_entries):
            if DENY_PARTS and any(p in (entry.get("link", "") or "").lower() for p in DENY_PARTS):
                continue
            if dup_mask[idx]:
                continue
            emb_vec = candidate_embs[idx]
            item = build_item(entry, emb_vec)
            # insert at top: position 0 within channel children after feedmeta if present
            channel.insert(0, item)
            # update caches
            if existing_normed.size == 0:
                existing_normed = normalize_matrix(np.expand_dims(emb_vec.astype(np.float32), axis=0))
            else:
                # insert new normalized vector at top
                new_norm = normalize_matrix(np.expand_dims(emb_vec.astype(np.float32), axis=0))
                existing_normed = np.vstack([new_norm, existing_normed])[:MAX_EXISTING]
            added_any = True

        # update last seen to newest entry's link
        set_feed_lastseen(channel, feed_url, entries[0].get("link", ""))

    # enforce MAX_ITEMS cap (keep newest at top)
    all_items = channel.findall("item")
    if len(all_items) > MAX_ITEMS:
        for it in all_items[MAX_ITEMS:]:
            channel.remove(it)

    tree.write(RESULT_FILE, encoding="utf-8", xml_declaration=True)

# -------------------- run once (this is final optimized single-run) --------------------
if __name__ == "__main__":
    process_once()