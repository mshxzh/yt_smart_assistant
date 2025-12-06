import os
import re
import tempfile
import json
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Optional: YouTube fetch (only if googleapiclient installed and YOUTUBE_API_KEY provided)
try:
    from googleapiclient.discovery import build
    _youtube_available = True
except Exception:
    build = None
    _youtube_available = False

# ---------- Config ----------
# Russian sentiment model (positive, neutral, negative)
RU_MODEL = "blanchefort/rubert-base-cased-sentiment"
RU_LABELS = ["neutral", "positive", "negative"]

# English/Other sentiment model (negative, neutral, positive)
EN_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EN_LABELS = ["negative", "neutral", "positive"]

# Unified sentiment labels for display
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

MAX_COMMENTS = 500

# ---------- Init models (lazy) ----------
_ru_tokenizer = None
_ru_model = None
_en_tokenizer = None
_en_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_ = load_dotenv(find_dotenv())  # read local .env file

# Language detection
from langdetect import detect, LangDetectException

def detect_language(text: str) -> str:
    """Detect language of text. Returns 'ru', 'en', or 'other'."""
    try:
        lang = detect(text)
        if lang == 'ru':
            return 'ru'
        elif lang == 'en':
            return 'en'
        else:
            return 'other'  # Use multilingual model for other languages
    except LangDetectException:
        return 'en'  # Default to English on detection failure


def init_ru_model():
    global _ru_tokenizer, _ru_model
    if _ru_tokenizer is None or _ru_model is None:
        _ru_tokenizer = AutoTokenizer.from_pretrained(RU_MODEL)
        _ru_model = AutoModelForSequenceClassification.from_pretrained(RU_MODEL).to(_device)


def init_en_model():
    global _en_tokenizer, _en_model
    if _en_tokenizer is None or _en_model is None:
        _en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL)
        _en_model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL).to(_device)


# ---------- YouTube fetcher ----------
def fetch_top_comments_from_youtube(video_id: str, max_comments: int = 200) -> List[dict]:
    """Fetch top-level comments using YouTube Data API v3.
       Requires environment variable YOUTUBE_API_KEY set (HF Spaces secret).
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key or not _youtube_available or build is None:
        raise RuntimeError("YouTube API not available. Provide comments manually or set YOUTUBE_API_KEY.")
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            order="relevance",  # top comments
            pageToken=next_page_token
        ).execute()
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "id": item.get("id"),
                "text": s.get("textDisplay", "") if isinstance(s, dict) else "",
                "author": s.get("authorDisplayName") if isinstance(s, dict) else None,
                "published_at": s.get("publishedAt") if isinstance(s, dict) else None,
                "like_count": s.get("likeCount", 0) if isinstance(s, dict) else 0
            })
            if len(comments) >= max_comments:
                break
        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break
    return comments


# ---------- Preprocessing ----------
import html

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TIMESTAMP_LINK_RE = re.compile(r'<a[^>]*href="[^"]*(?:&amp;t=|&t=|\?t=)[^"]*"[^>]*>([^<]+)</a>', re.IGNORECASE)
HTML_TAG_RE = re.compile(r'<[^>]+>')

def clean_text_for_display(s: str) -> str:
    """Clean YouTube HTML for display - preserves timestamps as readable text."""
    if not s:
        return ""
    # Convert timestamp links to plain timestamps with clock emoji
    s = TIMESTAMP_LINK_RE.sub(r'⏱️\1', s)
    # Convert <br> tags to newlines
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
    # Remove other HTML tags
    s = HTML_TAG_RE.sub('', s)
    # Decode HTML entities (&amp; -> &, &#39; -> ', etc.)
    s = html.unescape(s)
    # Normalize whitespace but preserve newlines
    s = re.sub(r'[^\S\n]+', ' ', s)
    s = re.sub(r'\n+', '\n', s)
    return s.strip()

def clean_text(s: str) -> str:
    """Clean text for emotion analysis - removes URLs and HTML."""
    if not s:
        return ""
    # Convert timestamp links to just the timestamp text
    s = TIMESTAMP_LINK_RE.sub(r'\1', s)
    # Convert <br> to space
    s = re.sub(r'<br\s*/?>', ' ', s, flags=re.IGNORECASE)
    # Remove all HTML tags
    s = HTML_TAG_RE.sub('', s)
    # Decode HTML entities
    s = html.unescape(s)
    # Remove URLs
    s = URL_RE.sub('', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------- Sentiment inference ----------
def _predict_batch(texts: List[str], tokenizer, model, labels, batch_size: int = 32):
    """Internal function to predict sentiment for a batch of texts using a specific model."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            idx = int(p.argmax())
            results.append({"label": labels[idx], "score": float(p[idx]), "probs": p.tolist()})
    return results


def predict_sentiment(texts: List[str], batch_size: int = 32):
    """Predict sentiment (positive/neutral/negative) for texts, using appropriate model based on language."""
    
    # Group texts by detected language (Russian vs other)
    ru_indices, ru_texts = [], []
    en_indices, en_texts = [], []  # English model used for English + other languages
    
    for i, text in enumerate(texts):
        lang = detect_language(text)
        if lang == 'ru':
            ru_indices.append(i)
            ru_texts.append(text)
        else:
            # Use English model for English and all other languages
            en_indices.append(i)
            en_texts.append(text)
    
    # Process Russian texts
    ru_results = []
    if ru_texts:
        init_ru_model()
        ru_results = _predict_batch(ru_texts, _ru_tokenizer, _ru_model, RU_LABELS, batch_size)
        for r in ru_results:
            r["lang"] = "ru"
    
    # Process English/other texts
    en_results = []
    if en_texts:
        init_en_model()
        en_results = _predict_batch(en_texts, _en_tokenizer, _en_model, EN_LABELS, batch_size)
        for r in en_results:
            r["lang"] = "en"
    
    # Merge results back in original order
    results = [None] * len(texts)
    for idx, res in zip(ru_indices, ru_results):
        results[idx] = res
    for idx, res in zip(en_indices, en_results):
        results[idx] = res
    
    return results