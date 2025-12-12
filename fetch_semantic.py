#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import feedparser
import os
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# feeds
FEEDS = [
    "http://www.thedailystar.net/latest/rss/rss.xml",
    "https://tbsnews.net/top-news/rss.xml",
    "https://www.dhakatribune.com/feed/"
]

RESULT_XML = "result.xml"
CACHE_JSON = "cache.json"

MAX_TOTAL = 1000
MAX_PER_FEED = 50
MAX_EXIST = 50
SIM_THRESH = 0.88
BLOCK_PARTS = ("/sport/", "/sports/", "/entertainment/")
IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"')

# load semantic model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(txts):
    return model.encode(txts, convert_to_numpy=True)

# read cache
def load_cache():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"embeds": {}}

def save_cache(c):
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(c, f)

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

# extract existing embeddings
def load_existing(channel, cache):
    items = channel.findall("item")[:MAX_EXIST]
    titles, vecs = [], []
    for it in items:
        t = it.findtext("title","") or ""
        if t in cache["embeds"]:
            vec = np.array(cache["embeds"][t],dtype=np.float32)
        else:
            vec = embed_batch([t])[0]
            cache["embeds"][t] = vec.tolist()
        titles.append(t)
        vecs.append(vec)
    if vecs:
        mat = np.vstack(vecs).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    else:
        mat = np.zeros((0,model.get_sentence_embedding_dimension()), dtype=np.float32)
    return titles, mat

# extract image from entry
def find_image(entry):
    # check media_content
    mc = entry.get("media_content")
    if mc and isinstance(mc,list):
        url = mc[0].get("url")
        if url:
            return url
    # check media_thumbnail
    mt = entry.get("media_thumbnail")
    if mt and isinstance(mt,list):
        url = mt[0].get("url")
        if url:
            return url
    # HTML in summary
    html = entry.get("summary", "") or ""
    m = IMG_RE.search(html)
    if m:
        return m.group(1)
    # HTML in content
    cont = entry.get("content")
    if cont and isinstance(cont,list):
        html2 = cont[0].get("value","")
        m2 = IMG_RE.search(html2)
        if m2:
            return m2.group(1)
    return None

# build RSS item
def make_item(entry, img_url):
    it = ET.Element("item")
    ET.SubElement(it, "title").text = entry.get("title","")
    ET.SubElement(it, "link").text = entry.get("link","")
    ET.SubElement(it, "pubDate").text = entry.get("published", datetime.utcnow().isoformat())
    ET.SubElement(it, "guid").text = entry.get("id", entry.get("link",""))
    desc = entry.get("summary","") or ""
    ET.SubElement(it, "description").text = desc
    if img_url:
        enc = ET.SubElement(it, "enclosure")
        enc.set("url", img_url)
        enc.set("length","0")
        enc.set("type","image/jpeg")
    return it

# main aggregator
def run_once():
    tree, channel = load_xml()
    cache = load_cache()
    existing_titles, existing_vecs = load_existing(channel, cache)

    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:MAX_PER_FEED]

        if not entries:
            continue

        titles = [e.get("title","") for e in entries]
        cand_vecs = embed_batch(titles)
        cand_norm = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)

        if existing_vecs.size>0:
            sims = cand_norm @ existing_vecs.T
            dup_mask = sims.max(axis=1) >= SIM_THRESH
        else:
            dup_mask = np.array([False]*len(entries))

        for i, entry in enumerate(entries):
            link = entry.get("link","").lower()
            if any(b in link for b in BLOCK_PARTS):
                continue
            if dup_mask[i]:
                continue
            img = find_image(entry)
            item = make_item(entry, img)
            channel.insert(3, item)
            t = entry.get("title","")
            cache["embeds"][t] = cand_vecs[i].tolist()

    items = channel.findall("item")
    if len(items)>MAX_TOTAL:
        for it in items[MAX_TOTAL:]:
            channel.remove(it)

    new_cache = {"embeds": {}}
    for it in channel.findall("item")[:MAX_EXIST]:
        t2 = it.findtext("title","")
        if t2 in cache["embeds"]:
            new_cache["embeds"][t2] = cache["embeds"][t2]
    save_cache(new_cache)

    tree.write(RESULT_XML,"utf-8",xml_declaration=True)

if __name__=="__main__":
    run_once()