# YouTube Smart Assistant 🎬

A powerful Streamlit application for analyzing YouTube videos through subtitle summarization, comment sentiment analysis, and comment clustering using AI.

## Features ✨

### 📝 Video Summarization
- Extract and process YouTube video subtitles/captions
- Generate AI-powered summaries using Google's Gemini
- Support for multiple subtitle languages
- Translate summaries to English, Dutch, or Russian

### 💬 Comment Sentiment Analysis
- Fetch and analyze YouTube comments
- Multilingual sentiment detection (English & Russian)
- Visual sentiment distribution (pie chart)
- Filter comments by sentiment (positive, neutral, negative)
- Language detection with flag indicators

### 🔮 Comment Clustering
- Group similar comments using HDBSCAN clustering
- LaBSE embeddings for multilingual support
- Extract top keywords per cluster using TF-IDF
- Interactive UMAP visualization
- Adjustable clustering parameters

### 🚀 Smart Features
- **Comment Caching**: Fetched comments are cached per video - no re-downloading when switching between analysis tabs
- **Incremental Fetching**: Request more comments and only the additional ones are fetched
- **Password Protection**: Secure access to the application

## Requirements 📋

- Python 3.10+
- Gemini API key (for summarization)
- YouTube Data API key (for comments)
- Internet connection

## Installation 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/mshxzh/yt_smart_assistant.git
   cd yt_smart_assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   YOUTUBE_API_KEY=your_youtube_api_key
   APP_PASSWORD=your_access_password
   ```

## Usage 💡

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Enter the access password

3. Paste a YouTube video URL

4. Choose your analysis:
   - **Summary Tab**: Generate AI summary from subtitles
   - **Sentiment Analysis Tab**: Analyze comment emotions
   - **Comment Clusters Tab**: Discover comment topics

## Project Structure 📁

```
video_summary_tool/
├── app.py                      # Main Streamlit application
├── src/
│   ├── utils.py               # Utilities, caching, styling
│   ├── llm_actions.py         # Gemini API interactions
│   ├── media_processing.py    # YouTube subtitle extraction
│   ├── comments_classification.py  # Sentiment analysis
│   └── comments_clustering.py      # HDBSCAN clustering
├── requirements.txt
└── README.md
```

## API Keys 🔑

### Gemini API
Get your key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)

### YouTube Data API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable YouTube Data API v3
3. Create credentials (API key)

## Technical Details 🔧

### Models Used
- **Summarization**: Google Gemini 2.0 Flash
- **Sentiment (English)**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Sentiment (Russian)**: `blanchefort/rubert-base-cased-sentiment`
- **Embeddings**: `sentence-transformers/LaBSE` (multilingual)
- **Clustering**: HDBSCAN with UMAP visualization

### Comment Caching
Comments are cached per video session:
- First fetch: Downloads and caches comments
- Subsequent requests: Uses cache if sufficient, fetches only additional needed
- Cache clears automatically when video URL changes

## Limitations 📝

- **Subtitles Required**: Summarization only works with videos that have captions
- **Comment Access**: Some videos have comments disabled
- **API Quotas**: YouTube API has daily quota limits
- **Processing Time**: Clustering large comment sets may take time

## Troubleshooting 🚨

| Issue | Solution |
|-------|----------|
| No subtitles available | Choose a video with captions enabled |
| Comments not loading | Check YOUTUBE_API_KEY in .env |
| Clustering fails | Try with more comments or lower min_cluster_size |
| SSL errors (macOS) | App includes fallback for NLTK stopwords |

## Supported URL Formats 🔗

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

## License 📄

For educational purposes only.

---

Built with ❤️ using Streamlit, Transformers, and Google Gemini
