from typing import Dict
import re
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import streamlit as st

# YouTube fetch
from googleapiclient.discovery import build


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def fetch_video_metadata(youtube_url: str) -> dict:
    """
    Fetch video metadata (title, description, tags, channel) from YouTube Data API.
    
    Args:
        youtube_url: YouTube video URL
        
    Returns:
        dict with video metadata
    """
    
    # Extract video ID
    match = re.search(r'(?:v=|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})', youtube_url)
    if not match:
        raise ValueError("Could not extract video ID from URL")
    video_id = match.group(1)
    
    if YOUTUBE_API_KEY:
        try:
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            
            response = youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            ).execute()
            
            if response.get("items"):
                snippet = response["items"][0]["snippet"]
                return {
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "tags": snippet.get("tags", []),
                    "published_at": snippet.get("publishedAt", ""),
                    "video_id": video_id
                }
        except Exception as e:
            print(f"YouTube API failed: {e}")

def find_captions(video_id: str) -> Dict[str, str]:
    """
    Finds all available captions for the video and returns a dictionary of language codes and names.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        Dict[str, str]: A dictionary containing the language codes and names of the available captions.
    """
    try:
        # Create API instance and list transcripts (v1.x API)
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        
        captions = {}
        # Get all available transcripts
        for transcript in transcript_list:
            lang_code = transcript.language_code
            lang_name = transcript.language
            
            captions[lang_code] = lang_name
        
        return captions
        
    except TranscriptsDisabled:
        return {}
    except NoTranscriptFound:
        return {}
    except VideoUnavailable:
        return {}
    except Exception as e:
        if "certificate verify failed" in str(e).lower():
            raise RuntimeError("SSL certificate verification failed while fetching captions. This is common on macOS. Please check your internet connection and try again.")
        else:
            raise RuntimeError(f"Error fetching captions: {str(e)}")


def retrieve_subtitles(video_id: str, selected_caption_language: str) -> str:
    """
    Retrieves the subtitles for the video in the preferred language.

    Args:
        video_id (str): The ID of the YouTube video.
        selected_caption_language (str): The language name (e.g., 'English', 'English (auto-generated)').

    Returns:
        str: The subtitles text.
    """
    try:
        # Create API instance and list transcripts (v1.x API)
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        
        # Find the matching transcript by language name
        target_transcript = None
        for transcript in transcript_list:
            lang_name = transcript.language
            if transcript.is_generated:
                lang_name = f"{lang_name} (auto-generated)"
            
            if lang_name == selected_caption_language:
                target_transcript = transcript
                break
        
        if target_transcript is None:
            return ""
        
        # Fetch the transcript data
        transcript_data = target_transcript.fetch()
        
        # Combine all text segments into a single string
        captions_text = " ".join([snippet.text for snippet in transcript_data])
        
        return captions_text

    except TranscriptsDisabled:
        print("Transcripts are disabled for this video")
        return ""
    except NoTranscriptFound:
        print("No transcript found for this video")
        return ""
    except VideoUnavailable:
        print("Video is unavailable")
        return ""
    except Exception as e:
        if "certificate verify failed" in str(e).lower():
            print(f"SSL certificate verification failed: {e}")
            raise RuntimeError("SSL certificate verification failed while retrieving subtitles. This is common on macOS. Please check your internet connection and try again.")
        else:
            print(f"Error retrieving subtitles: {e}")
            return ""

def fetch_comments_with_token(video_id: str, max_comments: int, page_token: str = None) -> tuple:
    """
    Fetch comments from YouTube API with pagination support.
    
    Returns:
        Tuple of (comments_list, next_page_token)
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = page_token
    
    while len(comments) < max_comments:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            order="relevance",
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
    
    return comments, next_page_token

def get_cached_comments(video_id: str, max_comments: int) -> list:
    """
    Get comments with caching. Only fetches additional comments if needed.
    
    Args:
        video_id: YouTube video ID
        max_comments: Maximum number of comments to return
        
    Returns:
        List of comment dictionaries
    """
    cache = st.session_state.comments_cache
    
    # If cache exists for this video and has enough comments, return from cache
    if cache and cache.get("video_id") == video_id:
        cached_comments = cache.get("comments", [])
        cached_count = len(cached_comments)
        
        # If we have enough cached comments, return them (no API call needed)
        if cached_count >= max_comments:
            st.info(f"✅ Using {max_comments} comments from cache ({cached_count} total cached)")
            return cached_comments[:max_comments]
        
        # If no more comments available (next_page_token is exhausted), return what we have
        if cache.get("next_page_token") is None and cached_count > 0:
            st.warning(f"⚠️ Only {cached_count} comments available for this video (requested {max_comments})")
            return cached_comments
        
        # Need to fetch more comments - fetch only the additional ones needed
        additional_needed = max_comments - cached_count
        st.info(f"📥 Fetching {additional_needed} additional comments (already have {cached_count} cached)")
        
        new_comments, next_token = fetch_comments_with_token(
            video_id, 
            additional_needed, 
            cache.get("next_page_token")
        )
        
        # Update cache
        all_comments = cached_comments + new_comments
        st.session_state.comments_cache = {
            "video_id": video_id,
            "comments": all_comments,
            "next_page_token": next_token
        }
        
        # Check if we got fewer than requested (video has limited comments)
        if len(all_comments) < max_comments and next_token is None:
            st.warning(f"⚠️ Only {len(all_comments)} comments available for this video")
        
        return all_comments[:max_comments]
    
    # No cache or different video - fetch fresh
    st.info(f"📥 Fetching {max_comments} comments...")
    comments, next_token = fetch_comments_with_token(video_id, max_comments, None)
    
    # Store in cache
    st.session_state.comments_cache = {
        "video_id": video_id,
        "comments": comments,
        "next_page_token": next_token
    }
    
    # Check if video has fewer comments than requested
    if len(comments) < max_comments and next_token is None:
        st.warning(f"⚠️ Only {len(comments)} comments available for this video")
    
    return comments

