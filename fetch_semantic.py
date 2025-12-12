#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import feedparser
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

FEEDS = [
    "http://www.thedailystar.net/latest/rss/rss.xml",
    "https://tbsnews.net/top-news/rss.xml",
    "https://www.dhakatribune.com/feed/"
]

RESULT_FILE = "result.xml"
CACHE_FILE = "cache.json"

MAX_ITEMS = 1000
MAX_FEED_ITEMS = 50
MAX_EXISTING = 50
SIM_THRESHOLD = 0.88

BLOCK = ("/sport/", "/sports/", "/entertainment/")

# ----------------------------------------
# Load model once
# ----------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed_batch(titles):
    return model.encode(titles, convert_to_numpy=True)

# ----------------------------------------
# Cache format:
# { "embeddings": { "title_text": [0.12, 0.44, ...] } }
# Only store last 50 embeddings.
# ----------------------------------------
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {"embeddings": {}}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

# ----------------------------------------
# XML helpers
# ----------------------------------------
def ensure_xml():
    if not os.path.exists(RESULT_FILE):
        root = ET.Element("rss")
        root.set("version", "2.0")
        ch = ET.SubElement(root, "channel")

        ET.SubElement(ch, "title").text = "Merged Feed"
        ET.SubElement(ch, "link").text = "http://localhost/"
        ET.SubElement(ch, "description").text = "Aggregated feed."

        ET.ElementTree(root).write(RESULT_FILE, encoding="utf-8", xml_declaration=True)

def load_xml():
    tree = ET.parse(RESULT_FILE)
    root = tree.getroot()
    channel = root.find("channel")
    return tree, channel

# ----------------------------------------
# Prepare existing embeddings
# ----------------------------------------
def get_existing_vectors(channel, cache):
    items = channel.findall("item")[:MAX_EXISTING]
    vecs = []
    titles = []

    for it in items:
        t = it.find("title").text or ""
        if t in cache["embeddings"]:
            v = np.array(cache["embeddings"][t], dtype=np.float32)
        else:
            v = embed_batch([t])[0]
            cache["embeddings"][t] = v.tolist()

        titles.append(t)
        vecs.append(v)

    if vecs:
        m = np.vstack(vecs).astype(np.float32)
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
    else:
        m = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    return titles, m

# ----------------------------------------
# RSS item builder
# ----------------------------------------
def build_item(entry):
    it = ET.Element("item")
    ET.SubElement(it, "title").text = entry.get("title", "")
    ET.SubElement(it, "link").text = entry.get("link", "")
    ET.SubElement(it, "pubDate").text = entry.get("published", datetime.utcnow().isoformat())
    ET.SubElement(it, "guid").text = entry.get("id", entry.get("link", ""))
    ET.SubElement(it, "description").text = entry.get("summary", "") or entry.get("title", "")
    return it

# ----------------------------------------
# The main optimized logic
# ----------------------------------------
def main():
    ensure_xml()
    tree, channel = load_xml()
    cache = load_cache()

    existing_titles, existing_vecs = get_existing_vectors(channel, cache)

    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:MAX_FEED_ITEMS]  # hard limit

        if not entries:
            continue

        titles = [e.get("title", "") for e in entries]
        links = [e.get("link", "") for e in entries]

        # Filter out blocked links
        filtered = []
        for e in entries:
            link = e.get("link", "").lower()
            if not any(bad in link for bad in BLOCK):
                filtered.append(e)
        entries = filtered

        if not entries:
            continue

        # Compute embeddings for new entries
        cand_titles = [e.get("title", "") for e in entries]
        cand_vecs = embed_batch(cand_titles)

        cand_vecs_norm = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)

        # Duplicate check
        if existing_vecs.size > 0:
            sims = cand_vecs_norm @ existing_vecs.T
            dup_mask = sims.max(axis=1) >= SIM_THRESHOLD
        else:
            dup_mask = np.array([False] * len(entries))

        # Insert only non-duplicates
        for i, entry in enumerate(entries):
            if dup_mask[i]:
                continue

            item = build_item(entry)
            # insert at top
            channel.insert(3, item)  # after title/link/description

            t = entry.get("title", "")
            cache["embeddings"][t] = cand_vecs[i].tolist()

    # rotate
    all_items = channel.findall("item")
    if len(all_items) > MAX_ITEMS:
        for it in all_items[MAX_ITEMS:]:
            channel.remove(it)

    # only keep last 50 embeddings
    new_cache = {"embeddings": {}}
    for it in channel.findall("item")[:MAX_EXISTING]:
        t = it.find("title").text or ""
        if t in cache["embeddings"]:
            new_cache["embeddings"][t] = cache["embeddings"][t]

    save_cache(new_cache)
    tree.write(RESULT_FILE, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()