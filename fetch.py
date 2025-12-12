import feedparser
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
FEEDS = [
    "http://www.thedailystar.net/latest/rss/rss.xml",
    "https://tbsnews.net/top-news/rss.xml",
    "https://www.dhakatribune.com/feed/",
]

OUTFILE = "result.xml"
MAX_ITEMS = 1000
BLOCK = ["/sport/", "/sports/", "/entertainment/"]
SIM_THRESHOLD = 0.88  # semantic similarity threshold

# -----------------------------
# LOAD OR INITIALIZE XML
# -----------------------------
def load_existing():
    if not os.path.exists(OUTFILE):
        root = ET.Element("rss")
        chan = ET.SubElement(root, "channel")
        ET.ElementTree(root).write(OUTFILE, encoding="utf-8", xml_declaration=True)
    tree = ET.parse(OUTFILE)
    return tree, tree.getroot().find("channel")

# -----------------------------
# SEMANTIC MODEL
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_title(title):
    return model.encode(title, convert_to_numpy=True)

def title_similar_semantic(a_embed, b_embed, threshold=SIM_THRESHOLD):
    sim = cosine_similarity([a_embed], [b_embed])[0][0]
    return sim >= threshold

# -----------------------------
# CHECK EXISTING
# -----------------------------
def exists_semantic(existing_embeds, title):
    t_embed = embed_title(title)
    for e_embed in existing_embeds:
        if title_similar_semantic(t_embed, e_embed):
            return True
    return False

# -----------------------------
# BLOCK LINKS
# -----------------------------
def blocked(link):
    link = link.lower()
    return any(x in link for x in BLOCK)

# -----------------------------
# ADD ITEM AT TOP
# -----------------------------
def add_item(channel, entry):
    item = ET.Element("item")
    ET.SubElement(item, "title").text = entry.get("title", "")
    ET.SubElement(item, "link").text = entry.get("link", "")
    ET.SubElement(item, "pubDate").text = entry.get("published", datetime.utcnow().isoformat())

    # Insert at top
    children = list(channel)
    for child in children:
        channel.remove(child)
    channel.insert(0, item)
    for child in children:
        channel.append(child)

    # Enforce max items
    items = channel.findall("item")
    if len(items) > MAX_ITEMS:
        for old in items[MAX_ITEMS:]:
            channel.remove(old)

# -----------------------------
# FETCH ONCE
# -----------------------------
def fetch_once():
    tree, channel = load_existing()
    # Precompute embeddings of existing titles
    existing_titles = [item.findtext("title","") for item in channel.findall("item")]
    existing_embeds = [embed_title(t) for t in existing_titles]

    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            title = e.get("title", "")
            link = e.get("link", "")
            if not title or not link:
                continue
            if blocked(link):
                continue
            if exists_semantic(existing_embeds, title):
                continue
            add_item(channel, e)
    tree.write(OUTFILE, encoding="utf-8", xml_declaration=True)

# -----------------------------
# MAIN LOOP (EVERY 5 MINUTES)
# -----------------------------
if __name__ == "__main__":
    while True:
        fetch_once()
        time.sleep(300)