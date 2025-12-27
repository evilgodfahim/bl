# -*- coding: utf-8 -*-

import feedparser
import os
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np
from urllib.parse import urlparse
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
FEEDS = [
    "https://evilgodfahim.github.io/ds/todays_news.xml",
    "https://evilgodfahim.github.io/ep/articles.xml",
    "https://en.prothomalo.com/feed/",
    "https://www.dhakatribune.com/feed/",
    "http://www.thedailystar.net/latest/rss/rss.xml",
    "https://tbsnews.net/top-news/rss.xml"
]

RESULT_XML = "result.xml"
CACHE_JSON = "cache.json"

MAX_TOTAL = 1000
MAX_PER_FEED = 50
MAX_EXIST = 50
SIM_THRESH = 0.70
BLOCK_PARTS = ("/opinion/", "/editorial/", "/sports/", "/sport/", "/entertainment/", 
                "/showtime/", "/video/", "/business/", "/cricket/", "/football/", "/event/")
IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
DAYS_TO_KEEP = 7
REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS = 3

# Load semantic model once
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Semantic model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load semantic model: {e}")
    model = None


def embed_batch(txts):
    """Generate embeddings for a batch of texts"""
    if not model or not txts:
        return np.array([])
    try:
        return model.encode(txts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.array([])


def load_cache():
    """Load cache with validation and migration"""
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure required keys exist
                if "title_log" not in data:
                    data["title_log"] = {}
                if "embeds" not in data:
                    data["embeds"] = {}
                # Validate structure
                if not isinstance(data["title_log"], dict):
                    data["title_log"] = {}
                if not isinstance(data["embeds"], dict):
                    data["embeds"] = {}
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading cache, creating new: {e}")
    return {"embeds": {}, "title_log": {}}


def save_cache(c):
    """Save cache with error handling"""
    try:
        with open(CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=2)
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def clean_old_entries(cache):
    """Remove entries older than DAYS_TO_KEEP"""
    cutoff = (datetime.utcnow() - timedelta(days=DAYS_TO_KEEP)).isoformat()
    old_titles = [t for t, ts in cache["title_log"].items() if ts < cutoff]
    removed = 0
    for t in old_titles:
        cache["title_log"].pop(t, None)
        cache["embeds"].pop(t, None)
        removed += 1
    if removed:
        logger.info(f"Cleaned {removed} old entries from cache")


def ensure_xml():
    """Create XML file if it doesn't exist"""
    if not os.path.exists(RESULT_XML):
        try:
            root = ET.Element("rss", {"version": "2.0", "xmlns:media": "http://search.yahoo.com/mrss/"})
            ch = ET.SubElement(root, "channel")
            ET.SubElement(ch, "title").text = "Aggregated Feed"
            ET.SubElement(ch, "link").text = ""
            ET.SubElement(ch, "description").text = "Aggregated news feed"
            tree = ET.ElementTree(root)
            tree.write(RESULT_XML, encoding="utf-8", xml_declaration=True)
            logger.info(f"Created new XML file: {RESULT_XML}")
        except Exception as e:
            logger.error(f"Error creating XML file: {e}")


def load_xml():
    """Load XML with error handling"""
    ensure_xml()
    try:
        tree = ET.parse(RESULT_XML)
        ch = tree.getroot().find("channel")
        if ch is None:
            raise ValueError("No channel element found in RSS")
        return tree, ch
    except Exception as e:
        logger.error(f"Error loading XML, recreating: {e}")
        # Recreate file
        if os.path.exists(RESULT_XML):
            os.remove(RESULT_XML)
        ensure_xml()
        return load_xml()


def load_existing_from_log(cache):
    """Extract existing titles and embeddings from cache"""
    titles = list(cache["title_log"].keys())
    if not titles:
        return [], np.array([])
    
    embeds = []
    valid_titles = []
    for t in titles:
        if t in cache["embeds"]:
            try:
                emb = cache["embeds"][t]
                if isinstance(emb, list):
                    embeds.append(np.array(emb))
                    valid_titles.append(t)
            except Exception as e:
                logger.warning(f"Invalid embedding for title: {t[:50]}...")
    
    if embeds:
        return valid_titles, np.vstack(embeds)
    return [], np.array([])


def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common artifacts
    text = re.sub(r'&[a-z]+;', ' ', text)
    return text.strip()


def extract_image(entry):
    """Extract image from various possible fields in feed entry"""
    image_url = ""
    
    # Try media:content
    if hasattr(entry, 'media_content') and entry.media_content:
        try:
            for media in entry.media_content:
                url = media.get('url', '')
                if url:
                    image_url = url
                    logger.debug(f"Found image in media_content: {url}")
                    return image_url
        except (IndexError, AttributeError, KeyError) as e:
            logger.debug(f"Error reading media_content: {e}")
    
    # Try media:thumbnail
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        try:
            for thumb in entry.media_thumbnail:
                url = thumb.get('url', '')
                if url:
                    image_url = url
                    logger.debug(f"Found image in media_thumbnail: {url}")
                    return image_url
        except (IndexError, AttributeError, KeyError) as e:
            logger.debug(f"Error reading media_thumbnail: {e}")
    
    # Try links with type image
    if hasattr(entry, 'links') and entry.links:
        for link in entry.links:
            if link.get('type', '').startswith('image/') or link.get('rel') == 'enclosure':
                url = link.get('href', '')
                if url:
                    image_url = url
                    logger.debug(f"Found image in links: {url}")
                    return image_url
    
    # Try enclosures
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get('type', '').startswith('image/') or enc.get('href', '').lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                url = enc.get('href', '')
                if url:
                    image_url = url
                    logger.debug(f"Found image in enclosures: {url}")
                    return image_url
    
    # Try content or summary for img tags
    content_fields = ['content', 'summary', 'description', 'summary_detail']
    for field in content_fields:
        if hasattr(entry, field):
            content = getattr(entry, field)
            if isinstance(content, list):
                content = content[0].get('value', '') if content else ''
            elif isinstance(content, dict):
                content = content.get('value', '')
            
            if content:
                # Try to find img tag
                match = IMG_RE.search(str(content))
                if match:
                    image_url = match.group(1)
                    logger.debug(f"Found image in {field} HTML: {image_url}")
                    return image_url
    
    # Try looking for any attribute with 'image' in the name
    for attr in dir(entry):
        if 'image' in attr.lower() and not attr.startswith('_'):
            try:
                val = getattr(entry, attr)
                if isinstance(val, str) and val.startswith('http'):
                    image_url = val
                    logger.debug(f"Found image in attribute {attr}: {val}")
                    return image_url
                elif isinstance(val, dict) and 'href' in val:
                    image_url = val['href']
                    logger.debug(f"Found image in attribute {attr}['href']: {image_url}")
                    return image_url
            except:
                pass
    
    if not image_url:
        logger.debug("No image found for this entry")
    
    return image_url


def extract_description(entry):
    """Extract description from various possible fields"""
    # Try summary first (most common)
    if hasattr(entry, 'summary') and entry.summary:
        return clean_text(entry.summary)
    
    # Try description
    if hasattr(entry, 'description') and entry.description:
        return clean_text(entry.description)
    
    # Try content
    if hasattr(entry, 'content') and entry.content:
        try:
            if isinstance(entry.content, list) and entry.content:
                return clean_text(entry.content[0].get('value', ''))
            elif isinstance(entry.content, dict):
                return clean_text(entry.content.get('value', ''))
        except (IndexError, AttributeError, KeyError):
            pass
    
    return ""


def parse_date(entry):
    """Parse publication date from entry with fallback options"""
    date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
    
    for field in date_fields:
        if hasattr(entry, field):
            date_tuple = getattr(entry, field)
            if date_tuple:
                try:
                    return datetime(*date_tuple[:6]).isoformat()
                except (TypeError, ValueError):
                    pass
    
    # Try string date fields
    string_fields = ['published', 'updated', 'created']
    for field in string_fields:
        if hasattr(entry, field):
            date_str = getattr(entry, field)
            if date_str:
                try:
                    # feedparser usually handles this, but just in case
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00')).isoformat()
                except (ValueError, AttributeError):
                    pass
    
    # Default to current time
    return datetime.utcnow().isoformat()


def fetch_feed(url, attempt=1):
    """Fetch and parse feed with retry logic"""
    try:
        logger.info(f"Fetching feed (attempt {attempt}): {url}")
        # Set user agent to avoid blocking
        feedparser.USER_AGENT = "Mozilla/5.0 (compatible; NewsAggregator/1.0)"
        
        # Parse without timeout parameter (not supported in older feedparser versions)
        feed = feedparser.parse(url)
        
        # Check for errors
        if hasattr(feed, 'bozo') and feed.bozo:
            logger.warning(f"Feed parse warning for {url}: {feed.get('bozo_exception', 'Unknown error')}")
        
        # Check if feed object is valid
        if not feed:
            logger.error(f"Failed to parse feed: {url}")
            return []
        
        # Validate feed has entries
        if not hasattr(feed, 'entries') or not feed.entries:
            logger.warning(f"No entries found in feed: {url}")
            return []
        
        logger.info(f"Successfully fetched {len(feed.entries)} entries from {url}")
        return feed.entries
        
    except Exception as e:
        logger.error(f"Error fetching feed {url} (attempt {attempt}): {e}")
        if attempt < RETRY_ATTEMPTS:
            time.sleep(2 ** attempt)  # Exponential backoff
            return fetch_feed(url, attempt + 1)
        return []


def should_block(link):
    """Check if article should be blocked based on URL"""
    if not link:
        return True
    return any(part in link.lower() for part in BLOCK_PARTS)


def is_duplicate(title, existing_titles, existing_embeds, cache):
    """Check if article is duplicate using semantic similarity"""
    if not title or not model:
        return True
    
    # Exact match check
    if title in existing_titles:
        logger.debug(f"Exact duplicate found: {title[:50]}...")
        return True
    
    # Semantic similarity check
    if len(existing_embeds) > 0:
        try:
            new_emb = embed_batch([title])[0]
            sims = np.dot(existing_embeds, new_emb) / (
                np.linalg.norm(existing_embeds, axis=1) * np.linalg.norm(new_emb)
            )
            max_sim = np.max(sims)
            if max_sim >= SIM_THRESH:
                logger.debug(f"Semantic duplicate found (sim={max_sim:.2f}): {title[:50]}...")
                return True
        except Exception as e:
            logger.error(f"Error checking similarity: {e}")
    
    return False


def add_article(channel, entry, cache, existing_titles, existing_embeds):
    """Add article to XML and update cache"""
    try:
        # Extract fields
        title = clean_text(entry.get('title', ''))
        link = entry.get('link', '')
        desc = extract_description(entry)
        pub_date = parse_date(entry)
        image = extract_image(entry)
        
        # Validation
        if not title or not link:
            logger.warning("Skipping entry without title or link")
            return False
        
        if should_block(link):
            logger.debug(f"Blocked article: {title[:50]}...")
            return False
        
        if is_duplicate(title, existing_titles, existing_embeds, cache):
            return False
        
        # Create item
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = title
        ET.SubElement(item, "link").text = link
        ET.SubElement(item, "description").text = desc or "No description available"
        ET.SubElement(item, "pubDate").text = pub_date
        
        # Add image if found
        if image:
            # Try multiple image elements for compatibility
            ET.SubElement(item, "enclosure", {"url": image, "type": "image/jpeg"})
            # Also add as media:thumbnail (some readers prefer this)
            img_elem = ET.SubElement(item, "{http://search.yahoo.com/mrss/}thumbnail")
            img_elem.set("url", image)
            logger.debug(f"Added image to article: {image}")
        else:
            logger.debug(f"No image found for article: {title[:50]}...")
        
        # Update cache
        cache["title_log"][title] = pub_date
        if model:
            try:
                emb = embed_batch([title])[0]
                cache["embeds"][title] = emb.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        
        logger.info(f"Added article: {title[:70]}...")
        return True
        
    except Exception as e:
        logger.error(f"Error adding article: {e}")
        return False


def main():
    """Main aggregation function"""
    logger.info("Starting RSS feed aggregation")
    
    # Load cache and clean old entries
    cache = load_cache()
    clean_old_entries(cache)
    
    # Load existing articles
    existing_titles, existing_embeds = load_existing_from_log(cache)
    logger.info(f"Loaded {len(existing_titles)} existing articles from cache")
    
    # Load XML
    tree, channel = load_xml()
    
    # Track statistics
    stats = {"total": 0, "added": 0, "blocked": 0, "duplicates": 0}
    
    # Process each feed
    for feed_url in FEEDS:
        entries = fetch_feed(feed_url)
        feed_added = 0
        
        for entry in entries[:MAX_PER_FEED]:
            stats["total"] += 1
            
            if add_article(channel, entry, cache, existing_titles, existing_embeds):
                feed_added += 1
                stats["added"] += 1
                
                # Update existing titles/embeds for duplicate checking
                title = clean_text(entry.get('title', ''))
                if title and title not in existing_titles:
                    existing_titles.append(title)
                    if title in cache["embeds"]:
                        emb = np.array(cache["embeds"][title])
                        if len(existing_embeds) > 0:
                            existing_embeds = np.vstack([existing_embeds, emb])
                        else:
                            existing_embeds = emb.reshape(1, -1)
            
            # Stop if we've added enough from this feed
            if feed_added >= MAX_PER_FEED:
                break
        
        logger.info(f"Feed {feed_url}: added {feed_added} articles")
        time.sleep(0.5)  # Be nice to servers
    
    # Save results
    try:
        tree.write(RESULT_XML, encoding="utf-8", xml_declaration=True)
        logger.info(f"Saved aggregated feed to {RESULT_XML}")
    except Exception as e:
        logger.error(f"Error saving XML: {e}")
    
    save_cache(cache)
    
    # Print statistics
    logger.info(f"""
Aggregation complete:
- Total entries processed: {stats['total']}
- Articles added: {stats['added']}
- Total articles in cache: {len(cache['title_log'])}
    """)


if __name__ == "__main__":
    main()