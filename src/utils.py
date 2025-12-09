import base64
import streamlit as st


def get_base64(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

def convert_youtube_url(url):
    """
    Converts a standard YouTube video URL to an embeddable URL format.

    Standard YouTube URLs look like:
    'https://www.youtube.com/watch?v=<video_id>'
    or
    'https://youtu.be/<video_id>'

    To embed these videos, the URL should be in the format:
    'https://www.youtube.com/embed/<video_id>'

    Parameters:
        url (str): The original URL of the YouTube video.

    Returns:
        str: The embeddable URL for the YouTube video.
    """
    import re

    # Extract video ID using regex
    match = re.search(r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")

    video_id = match.group(1)

    # Construct the embeddable URL
    embed_url = f"https://www.youtube.com/embed/{video_id}"

    return embed_url

def style_css(background_image):
    return f"""
    <style>
        /* General styles */
        .small-text {{ font-size: 16px; color: #ffffff; margin-bottom: -10px;}}
         
        /* Header styling */
        header[data-testid="stHeader"] {{ background-color: rgba(0, 0, 0, 0.3); color: white; padding: 20px; }}
        /* Main content styling */
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image}");
            background-size: cover;
        }}
        /* Area styling */
        [data-testid="stExpander"] summary {{
            color: white !important;  /* Expander title */
        }}
        [data-testid="stSidebar"] .stTextInput input::placeholder {{
            color: rgba(255, 255, 255, 0.5) !important;
        }}
        .area-title {{
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 15px;
            margin-top: 30px;
            display: flex;
            align-items: center;
        }}
        
        /* Apple Liquid Glass Title Effect */
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800&display=swap');
        
        .liquid-glass-title {{
            font-family: 'Nunito', sans-serif;
            font-size: 48px;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(
                135deg,
                rgba(255, 255, 255, 1) 0%,
                rgba(240, 245, 255, 1) 25%,
                rgba(235, 240, 255, 1) 50%,
                rgba(245, 240, 255, 1) 75%,
                rgba(255, 255, 255, 1) 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 2px 4px rgba(255, 255, 255, 0.4))
                    drop-shadow(0 0 15px rgba(255, 255, 255, 0.1));
            letter-spacing: -0.5px;
            margin: 0;
            padding: 10px 0;
        }}
        
    </style>
    
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <p class="liquid-glass-title">Youtube Smart Assistant</p>
    <div class="small-text"><p style="text-align:center;">Get summaries from YouTube video subtitles, get sentiment analysis and cluster comments.</p></div>
    <br>
    <div class="small-text"><p style="text-align:center; font-size:15px; font-weight: bold">Disclaimer: for educational purposes only!</p></div>
"""

def display_cache_status():
    """Display current cache status in the UI."""
    cache = st.session_state.comments_cache
    if cache:
        cached_count = len(cache.get("comments", []))
        has_more = cache.get("next_page_token") is not None
        more_text = " (more available)" if has_more else " (all fetched)"
        st.caption(f"💾 Cache: {cached_count} comments{more_text}")
    else:
        st.caption("💾 Cache: empty")
