# YouTube Smart Assistant 🎬

An app for summarizing YouTube videos, analyzing comments, and clustering discussions.

Two options are available (choose based on deployment setting):

- **app_captions.py** (best accuracy, subtitles-based) — For local/educational use only; it processes video captions to produce highly accurate summaries.
- **app_metadata.py** (cloud-safe, metadata-based) — For cloud deployments where pulling full YouTube content is a concern; uses only public metadata (title/description), so summaries are faster but less precise.

## Features ✨

### 📝 Video Summarization
- **Captions-based** (`app_captions.py`): Extract and process YouTube video subtitles/captions
- **Metadata-based** (`app_metadata.py`): Use video title and description for quick summaries
- Generate AI-powered summaries using Google's Gemini 2.5 Flash
- Support for multiple subtitle languages (captions app)
- Translate summaries to English, Dutch, Russian, or use detected language
- Rich video metadata display (title, channel, publish date, description)

### 💬 Comment Sentiment Analysis
- Fetch and analyze YouTube comments using YouTube Data API
- Multilingual sentiment detection (English & Russian)
- Visual sentiment distribution with interactive pie charts
- Filter comments by sentiment (positive, neutral, negative)
- Language detection with flag indicators (🇬🇧 🇷🇺)
- Detailed sentiment scores and confidence levels
- Color-coded comment cards for easy scanning

### 🔮 Comment Clustering
- Group similar comments using HDBSCAN clustering algorithm
- LaBSE embeddings for multilingual comment support
- Extract top keywords per cluster using TF-IDF with NLTK stopwords
- Interactive UMAP 2D visualization
- Adjustable clustering parameters (min cluster size, max comments)
- Cluster size distribution charts
- Keyword tags with color coding per cluster

### 🚀 Smart Features
- **Comment Caching**: Fetched comments are cached per video - no re-downloading when switching between analysis tabs
- **Incremental Fetching**: Request more comments and only the additional ones are fetched (e.g., 50 → 100 only fetches 50 more)
- **Video Metadata Display**: Beautiful card showing video title, channel, publish date, and description preview
- **Error Handling**: Graceful fallbacks for missing captions, SSL issues, and API failures

## Requirements 📋
- Python 3.10+
- `GEMINI_API_KEY` (summaries)
- `YOUTUBE_API_KEY` (comments + metadata)
- Internet connection

## Installation 🚀
```bash
git clone https://github.com/mshxzh/yt_smart_assistant.git
cd yt_smart_assistant
pip install -r requirements.txt
```

Create `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
YOUTUBE_API_KEY=your_youtube_api_key
```

## Run the apps 💡
- Captions (preferred for accurate summaries):
  ```bash
  streamlit run app_captions.py
  ```
- Metadata (quick, works without captions):
  ```bash
  streamlit run app_metadata.py
  ```

Then:
1) Enter the access password  
2) Paste a YouTube URL  
3) Generate summary, run sentiment analysis, or cluster comments

## Project Structure 📁
```
video_summary_tool/
├── app_captions.py        # Captions-based summarizer (accurate)
├── app_metadata.py        # Metadata-based summarizer (fast fallback)
├── src/
│   ├── utils.py
│   ├── llm_actions.py     # Gemini interactions
│   ├── media_processing.py # YouTube metadata, captions, comments
│   ├── comments_classification.py
│   └── comments_clustering.py
├── requirements.txt
└── README.md
```

## Technical Details 🔧
- Summarization LLM: Gemini 2.5 Flash
- Embeddings: `sentence-transformers/LaBSE`
- Clustering: HDBSCAN + UMAP
- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest` (EN), `blanchefort/rubert-base-cased-sentiment` (RU)
- Comment caching: per-video, incremental, auto-cleared on URL change

## Limitations 📝
- Captions app: needs videos with subtitles for accurate summaries
- Metadata app: faster but less precise (no transcript)
- Comments may be disabled or limited; YouTube API quotas apply

## Troubleshooting 🚨
- No subtitles → use a video with captions or switch to `app_metadata.py`
- Comments not loading → check `YOUTUBE_API_KEY`
- Clustering fails → increase comments or reduce `min_cluster_size`
- SSL issues on macOS (NLTK) → fallback stopwords already included

## Supported URL Formats 🔗
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

---

Built with ❤️ using Streamlit, Transformers, and Google Gemini
