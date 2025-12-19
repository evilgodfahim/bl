# -*- coding: utf-8 -*-

import feedparser
import os
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np

# feeds
FEEDS = [
    "http://www.thedailystar.net/latest/rss/rss.xml",
    "https://tbsnews.net/top-news/rss.xml",
    "https://www.dhakatribune.com/feed/",
    "https://evilgodfahim.github.io/ds/todays_news.xml"
]

RESULT_XML = "result.xml"
CACHE_JSON = "cache.json"

MAX_TOTAL = 1000
MAX_PER_FEED = 50
MAX_EXIST = 50
SIM_THRESH = 0.75
BLOCK_PARTS = ("/sport/", "/sports/", "/entertainment/", "/videos/", "/video/", "/showtime/")
IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"')
DAYS_TO_KEEP = 7

# load semantic model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(txts):
    return model.encode(txts, convert_to_numpy=True, show_progress_bar=False)

# read cache
def load_cache():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "title_log" not in data:
                data["title_log"] = {}
            if "embeds" not in data:
                data["embeds"] = {}
            return data
    return {"embeds": {}, "title_log": {}}

def save_cache(c):
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(c, f)

def clean_old_entries(cache):
    cutoff = (datetime.utcnow() - timedelta(days=DAYS_TO_KEEP)).isoformat()
    old_titles = [t for t, ts in cache["title_log"].items() if ts < cutoff]
    for t in old_titles:
        cache["title_log"].pop(t, None)
        cache["embeds"].pop(t, None)

# ensure XML skeleton
def ensure_xml():
    if not os.path.exists(RESULT_XML):
        root = ET.Element("rss", {"version":"2.0"})
        ch = ET.SubElement(root, "channel")
        ET.SubElement(ch, "title").text = "Aggregated Feed"
        ET.SubElement(ch, "link").text = ""
        ET.SubElement(ch, "description").text = ""
        ET.ElementTree(root).write(RESULT_XML, "utf-8", xml_declaration=True)

def load_xml():
    ensure_xml()
    tree = ET.parse(RESULT_XML)
    ch = tree.getroot().find("channel")
    return tree, ch

# extract existing embeddings from 7-day log
def load_existing_from_log(cache):
    titles = list(cache["title_log"].keys())
    if not titles:
        return titles, np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    vecs = []
    valid_titles = []

    for t in titles:
        if t in cache["embeds"]:
            vec = np.array(cache["embeds"][t], dtype=np.float32)
            vecs.append(vec)
            valid_titles.append(t)

    if vecs:
        mat = np.vstack(vecs).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    else:
        mat = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    return valid_titles, mat

# extract image from entry
def find_image(entry):
    # 1. Check enclosure first (RSS <enclosure>)
    enc = entry.get("enclosures")
    if enc and isinstance(enc, list) and "url" in enc[0]:
        return enc[0]["url"]

    # 2. Check media_content
    mc = entry.get("media_content")
    if mc and isinstance(mc, list):
        url = mc[0].get("url")
        if url:
            return url

    # 3. Check media_thumbnail
    mt = entry.get("media_thumbnail")
    if mt and isinstance(mt, list):
        url = mt[0].get("url")
        if url:
            return url

    # 4. Check <img> in summary
    html = entry.get("summary", "") or ""
    m = IMG_RE.search(html)
    if m:
        return m.group(1)

    # 5. Check <img> in content
    cont = entry.get("content")
    if cont and isinstance(cont, list):
        html2 = cont[0].get("value", "")
        m2 = IMG_RE.search(html2)
        if m2:
            return m2.group(1)

    return None

# build RSS item
def make_item(entry, img_url):
    it = ET.Element("item")
    ET.SubElement(it, "title").text = entry.get("title", "")
    ET.SubElement(it, "link").text = entry.get("link", "")
    ET.SubElement(it, "pubDate").text = entry.get("published", datetime.utcnow().isoformat())
    ET.SubElement(it, "guid").text = entry.get("id", entry.get("link", ""))
    desc = entry.get("summary", "") or ""
    ET.SubElement(it, "description").text = desc
    if img_url:
        enc = ET.SubElement(it, "enclosure")
        enc.set("url", img_url)
        enc.set("length", "0")
        enc.set("type", "image/jpeg")
    return it

# main aggregator
def run_once():
    tree, channel = load_xml()
    cache = load_cache()

    # Clean entries older than 7 days
    clean_old_entries(cache)

    # Load existing titles from 7-day log
    existing_titles, existing_vecs = load_existing_from_log(cache)

    current_time = datetime.utcnow().isoformat()
    new_items_added = 0

    # Collect ALL candidates from ALL feeds first
    all_candidates = []

    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:MAX_PER_FEED]

        if not entries:
            continue

        # Filter out blocked entries
        for entry in entries:
            link = entry.get("link", "").lower()
            if any(b in link for b in BLOCK_PARTS):
                continue
            title = entry.get("title", "")
            if not title:
                continue
            all_candidates.append(entry)

    if not all_candidates:
        save_cache(cache)
        tree.write(RESULT_XML, "utf-8", xml_declaration=True)
        print("No new candidates found.")
        return

    # Batch embed ALL candidates at once (single embedding call)
    all_titles = [e.get("title", "") for e in all_candidates]
    cand_vecs = embed_batch(all_titles)
    cand_norm = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)

    # Check against existing embeddings
    if existing_vecs.size > 0:
        sims_existing = cand_norm @ existing_vecs.T
        dup_mask = sims_existing.max(axis=1) >= SIM_THRESH
    else:
        dup_mask = np.array([False] * len(all_candidates))

    # Now process candidates and check for duplicates within the batch
    added_indices = []
    added_vecs = []

    for i, entry in enumerate(all_candidates):
        # Skip if duplicate with existing
        if dup_mask[i]:
            continue

        # Check against already added items in THIS run
        if added_vecs:
            added_matrix = np.vstack(added_vecs)
            sims_new = cand_norm[i:i+1] @ added_matrix.T
            if sims_new.max() >= SIM_THRESH:
                continue

        # Not a duplicate - add it
        title = all_titles[i]
        img = find_image(entry)
        item = make_item(entry, img)
        channel.insert(3, item)

        # Update cache with new title and embedding
        cache["embeds"][title] = cand_vecs[i].tolist()
        cache["title_log"][title] = current_time

        # Track for within-run duplicate checking
        added_indices.append(i)
        added_vecs.append(cand_norm[i])
        new_items_added += 1

    # Trim items if exceeding MAX_TOTAL
    items = channel.findall("item")
    if len(items) > MAX_TOTAL:
        for it in items[MAX_TOTAL:]:
            channel.remove(it)

    # Save cache and XML
    save_cache(cache)
    tree.write(RESULT_XML, "utf-8", xml_declaration=True)

    print(f"Run completed. Added {new_items_added} new articles. Total in log: {len(cache['title_log'])}")

if __name__ == "__main__":
    run_once()