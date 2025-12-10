import re
import time
import os

import streamlit as st
import plotly.express as px
import pandas as pd

from src.media_processing import fetch_video_metadata, get_cached_comments
from src.comments_classification import *
from src.comments_clustering import cluster_comments, reduce_dimensions_for_plot, top_keywords_per_cluster_nltk
from src.llm_actions import summarize_video_from_metadata
from src.utils import *

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
APP_PASSWORD = os.getenv("APP_PASSWORD")


# Set the page configuration (should be at the top)
st.set_page_config(page_title="Youtube Smart Assistant", layout='centered', page_icon=":material/subtitles:")

# Background image https://unsplash.com/photos/blue-and-yellow-abstract-painting-1xZ0SqLPE4E
background_image = get_base64('src/background.jpg')  # Replace with your light theme image

st.markdown(style_css(background_image), unsafe_allow_html=True)

st.divider() 

# ========== PASSWORD PROTECTION ==========
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    """Show password dialog and verify access."""
    if st.session_state.authenticated:
        return True
    
    # Create a centered container for the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style="background: rgba(0,0,0,0.4); padding: 15px 40px; border-radius: 12px; backdrop-filter: blur(10px); margin: 20px auto 0 auto; max-width: 500px;">
                <p style="color: white; text-align: center; margin-bottom: 8px; font-size: 30px; font-weight: bold;">🔐 Access Required</p>
                <p style="color: rgba(255,255,255,0.7); text-align: center; margin-bottom: 0; font-size: 14px;">Please enter the password to continue</p>
            </div>
        """, unsafe_allow_html=True)
        
        password = st.text_input("", type="password", key="password_input", placeholder="Enter password...")
        
        if st.button("🔓 Unlock", type="primary", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
        
        st.markdown("""
            <p style="color: rgba(255,255,255,0.5); text-align: center; font-size: 12px; margin-top: 20px;">
                Contact the administrator if you need access
            </p>
        """, unsafe_allow_html=True)
    
    return False

# Check password before showing main content
if not check_password():
    st.stop()

# ========== END PASSWORD PROTECTION ==========

# Initialize session state variables
if "summary" not in st.session_state:
    st.session_state.summary = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None
if "video_description" not in st.session_state:
    st.session_state.video_description = None
if "video_channel" not in st.session_state:
    st.session_state.video_channel = None
if "video_published_at" not in st.session_state:
    st.session_state.video_published_at = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "previous_url" not in st.session_state:
    st.session_state.previous_url = None
    st.session_state.disabled_button = False  # Default to not disabled
if "use_original_language" not in st.session_state:
    st.session_state.use_original_language = True  # Initialize language preference
if "comments_data" not in st.session_state:
    st.session_state.comments_data = None
if "cluster_data" not in st.session_state:
    st.session_state.cluster_data = None

if "youtube_key" not in st.session_state:
    st.session_state.youtube_key = 1

if "condition_yt" not in st.session_state:
    st.session_state.condition_yt = False

# Comments cache: stores fetched comments to avoid re-downloading
# Structure: {"video_id": str, "comments": list, "next_page_token": str|None}
if "comments_cache" not in st.session_state:
    st.session_state.comments_cache = None

def clear_outputs():
    st.session_state.video_title = None
    st.session_state.video_description = None
    st.session_state.video_channel = None
    st.session_state.video_published_at = None
    st.session_state.video_id = None
    st.session_state.summary = None
    st.session_state.comments_data = None
    st.session_state.cluster_data = None
    st.session_state.comments_cache = None  # Clear comments cache when video changes
    

def main():
    # Callback functions to update state and rerun UI
    def on_youtube_input():
        # Check if valid YouTube URL is provided
        youtube_pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        condition_yt = st.session_state[f"yt_{st.session_state.youtube_key}"] != '' and re.match(youtube_pattern, st.session_state[f"yt_{st.session_state.youtube_key}"]) is not None  
    
        st.session_state.condition_yt = condition_yt
    
    st.session_state.api_key_validated = True

    # YouTube URL input
    st.text_input(
        "YouTube Link:",
        placeholder="Enter a YouTube video link",
        key=f"yt_{st.session_state.youtube_key}",
        on_change=on_youtube_input
    )

    # Display video if valid URL is provided
    if st.session_state.condition_yt:
        # VIDEO section
        st.markdown('<div class="area-title">Your Video</div>', unsafe_allow_html=True)
        emb_youtube_url = convert_youtube_url(st.session_state[f"yt_{st.session_state.youtube_key}"])
        st.markdown(f'<iframe width="100%" height="450" src="{emb_youtube_url}" frameBorder="0" allow="clipboard-write; autoplay" webkitAllowFullScreen mozallowfullscreen allowFullScreen></iframe>', unsafe_allow_html=True)
        
        # Fetch and store video metadata
        video_metadata = fetch_video_metadata(st.session_state[f"yt_{st.session_state.youtube_key}"])
        st.session_state.video_title = video_metadata["title"]
        st.session_state.video_id = video_metadata["video_id"]
        st.session_state.video_description = video_metadata["description"]
        st.session_state.video_channel = video_metadata["channel"]
        st.session_state.video_published_at = video_metadata["published_at"]
        
        # Format published date
        published_date = ""
        if st.session_state.video_published_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(st.session_state.video_published_at.replace('Z', '+00:00'))
                published_date = dt.strftime("%b %d, %Y")
            except:
                published_date = st.session_state.video_published_at[:10] if st.session_state.video_published_at else ""
        
        # Truncate description for display
        description_preview = re.sub(r'\n', ' ', st.session_state.video_description) or ""
        if len(description_preview) > 200:
            description_preview = description_preview[:200] + "..."
        
        st.markdown(f'''
            <div style="background: rgba(0,0,0,0.3); padding: 15px 20px; border-radius: 12px; margin-bottom: 20px;">
                <p style="font-size: 18px; font-weight: bold; color: #ffffff; margin: 0 0 8px 0;">{st.session_state.video_title}</p>
                <p style="font-size: 14px; color: #1DB954; margin: 0 0 8px 0;">
                    📺 {st.session_state.video_channel} {f"• 📅 {published_date}" if published_date else ""}
                </p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0; line-height: 1.4;">{description_preview}</p>
            </div>
        ''', unsafe_allow_html=True)

    # Check if previous URL changed to clear outputs
    # Only clear if previous_url is set AND different from current URL
    current_url = st.session_state[f"yt_{st.session_state.youtube_key}"] if st.session_state.condition_yt else None
    has_any_data = st.session_state.summary is not None or st.session_state.comments_data is not None or st.session_state.cluster_data is not None
    
    if has_any_data and st.session_state.previous_url is not None:
        if current_url and st.session_state.previous_url != current_url:
            clear_outputs()

    if st.session_state.condition_yt:
        # Create tabs for Summary, Sentiment Analysis, and Clustering
        tab_summary, tab_comments, tab_clusters = st.tabs(["📝 Summary", "💬 Sentiment Analysis", "🔮 Comment Clusters"])
        
        # ---------- SUMMARY TAB ----------
        with tab_summary:

            # Language settings
            with st.expander("Summary language settings"):
                col1, col2 = st.columns([1,1])
                with col1:
                    st.session_state.use_original_language = st.toggle(
                        "Use detected language", 
                        value=st.session_state.use_original_language,
                    )
                with col2:
                    lang_option = st.selectbox(
                        "What language do you prefer?",
                        ("English", "Dutch", "Russian"),
                        disabled=st.session_state.use_original_language # or st.session_state.language_settings_disabled,
                    )
            if st.session_state.use_original_language:
                lang_option = ""
            
            st.markdown('<p style="margin-bottom: 30px;"></p>', unsafe_allow_html=True)

            # Button to analyze subtitle content
            generate_content_button = st.button("Get Video Description", 
                    type='primary', 
                    use_container_width=True, 
                    key="summary_btn")

            if generate_content_button:
                start_time = time.time()
                
                # ===== Direct URL-based summary using LLM =====
                with st.spinner("Analyzing video with AI..."):
                    try:
                        video_url = st.session_state[f"yt_{st.session_state.youtube_key}"]
                        summary = summarize_video_from_metadata(video_metadata, chosen_language=lang_option)
                        
                        st.session_state.summary = summary
                        st.session_state.previous_url = video_url

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        st.info(f"Time taken: {elapsed_time:.2f} seconds", icon=":material/timer:")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

            if st.session_state.summary:
                with st.expander("Summary", expanded=True):
                    col1, col2 = st.columns([2,1])
                    with col1:
                        if st.button("Re-generate Summary", type='secondary', key="regen_btn"):
                            # Re-generate using URL-based approach
                            with st.spinner("Re-analyzing video..."):
                                video_url = st.session_state[f"yt_{st.session_state.youtube_key}"]
                                summary = summarize_video_from_metadata(video_metadata, chosen_language=lang_option)
                                st.session_state.summary = summary

                    st.markdown(f'''
                        <div style="margin-bottom: 15px;">
                            <p style="font-size: 16px; line-height: 1.5; color: #ffffff;">{st.session_state.summary}</p>
                        </div>
                    ''', unsafe_allow_html=True)
        
        # ---------- SENTIMENT ANALYSIS TAB ----------
        with tab_comments:
            st.markdown('<p style="margin-bottom: 15px;"></p>', unsafe_allow_html=True)
            
            # Settings for sentiment analysis
            with st.expander("Analysis Settings", expanded=True):
                max_comments = st.slider("Maximum comments to analyze", min_value=10, max_value=500, value=50, step=10)
                display_cache_status()
            
            # Check if API key is available for YouTube
            analyze_sentiment_button = st.button("Analyze Sentiment", 
                    type='primary', 
                    use_container_width=True, 
                    key="sentiment_btn")

            # Sentiment labels with colors and emojis
            SENTIMENT_STYLES = {
                "positive": {"color": "#43A047", "emoji": "👍"},
                "neutral": {"color": "#9E9E9E", "emoji": "😐"},
                "negative": {"color": "#E53935", "emoji": "👎"},
            }

            if analyze_sentiment_button:
                if not st.session_state.video_id:
                    st.error("Could not extract video ID from URL")
                else:
                    with st.spinner("Fetching and analyzing comments..."):
                        try:
                            # Fetch comments (with caching)
                            comments = get_cached_comments(st.session_state.video_id, max_comments=max_comments)
                            
                            if not comments:
                                st.warning("No comments found for this video.")
                            else:
                                st.success(f"Using {len(comments)} comments")
                                
                                # Clean texts: one for analysis, one for display
                                raw_texts = [c.get("text", "") for c in comments]
                                analysis_texts = [clean_text(t) for t in raw_texts]
                                display_texts = [clean_text_for_display(t) for t in raw_texts]
                                
                                # Filter out empty texts (keep indices aligned)
                                valid_data = [(disp, anal) for disp, anal in zip(display_texts, analysis_texts) if anal.strip()]
                                
                                if valid_data:
                                    display_texts, analysis_texts = zip(*valid_data)
                                    sentiments = predict_sentiment(list(analysis_texts))
                                    
                                    # Store display text with sentiments in session state
                                    st.session_state.comments_data = list(zip(display_texts, sentiments))
                                    # Track URL to detect video changes
                                    st.session_state.previous_url = st.session_state[f"yt_{st.session_state.youtube_key}"]
                                    
                        except Exception as e:
                            st.error(f"Error fetching comments: {str(e)}")
                            st.info("Make sure YOUTUBE_API_KEY is set in your environment variables.")
            
            # Display results if available
            if "comments_data" in st.session_state and st.session_state.comments_data:
                comments_data = st.session_state.comments_data
                total = len(comments_data)
                
                # Calculate emotion distribution
                emotion_counts = {}
                for _, e in comments_data:
                    label = e["label"]
                    emotion_counts[label] = emotion_counts.get(label, 0) + 1
                
                # Only show sentiments that were detected (count > 0), sorted by count
                detected_sentiments = [(e, c) for e, c in emotion_counts.items() if c > 0]
                detected_sentiments.sort(key=lambda x: x[1], reverse=True)
                
                # Display pie chart and table
                st.markdown("### Sentiment Distribution")
                
                col_chart, col_table = st.columns([2, 1])
                
                with col_chart:
                    # Prepare data for pie chart
                    labels = [f"{SENTIMENT_STYLES.get(e, {}).get('emoji', '❓')} {e.capitalize()}" for e, _ in detected_sentiments]
                    values = [c for _, c in detected_sentiments]
                    colors = [SENTIMENT_STYLES.get(e, {"color": "#9E9E9E"})["color"] for e, _ in detected_sentiments]
                    
                    fig = px.pie(
                        values=values,
                        names=labels,
                        color_discrete_sequence=colors,
                        hole=0.4  # Donut chart
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
                    )
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.3,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=10)
                        ),
                        margin=dict(t=20, b=20, l=20, r=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_table:
                    # Create table data
                    table_data = {
                        "Sentiment": [f"{SENTIMENT_STYLES.get(e, {}).get('emoji', '❓')} {e.capitalize()}" for e, _ in detected_sentiments],
                        "#": [c for _, c in detected_sentiments],
                        "%": [f"{100*c/total:.1f}%" for _, c in detected_sentiments]
                    }
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, hide_index=True, use_container_width=True)
                
                # Display comments with sentiment
                st.markdown("### Analyzed Comments")
                
                # Filter options - only show detected sentiments
                filter_options = ["All"] + [e.capitalize() for e, _ in detected_sentiments]
                filter_sentiment = st.selectbox(
                    "Filter by sentiment:",
                    filter_options,
                    key="filter_sentiment"
                )
                
                import html as html_module
                
                for text, sentiment in comments_data:
                    if filter_sentiment != "All" and sentiment["label"].lower() != filter_sentiment.lower():
                        continue
                    
                    # Get style for this sentiment
                    style = SENTIMENT_STYLES.get(sentiment["label"], {"color": "#9E9E9E", "emoji": "❓"})
                    color = style["color"]
                    emoji = style["emoji"]
                    lang = sentiment.get("lang", "")
                    lang_flag = "🇷🇺" if lang == "ru" else "🇬🇧"
                    
                    # Escape HTML and convert newlines to <br> for display
                    safe_text = html_module.escape(text).replace('\n', '<br>')
                    
                    st.markdown(f'''
                        <div style="background-color: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {color};">
                            <p style="margin: 0; color: #ffffff; font-size: 14px; white-space: pre-wrap;">{safe_text}</p>
                            <p style="margin: 5px 0 0 0; color: {color}; font-size: 12px;">{lang_flag} {emoji} {sentiment["label"].capitalize()} ({sentiment["score"]:.2%})</p>
                        </div>
                    ''', unsafe_allow_html=True)
        
        # ---------- CLUSTERING TAB ----------
        with tab_clusters:
            st.markdown('<p style="margin-bottom: 15px;"></p>', unsafe_allow_html=True)
            
            # Settings for clustering
            with st.expander("Clustering Settings", expanded=True):
                max_comments_cluster = st.slider(
                    "Maximum comments to cluster", 
                    min_value=20, max_value=500, value=250, step=10,
                    key="cluster_max_comments"
                )
                min_cluster_size = st.slider(
                    "Min cluster size", 
                    min_value=3, max_value=10, value=3, step=1,
                    help="Minimum number of comments to form a cluster",
                    key="cluster_min_size"
                )
                display_cache_status()
            
            # Get values from session state to ensure they persist through button click
            current_max_comments = st.session_state.get("cluster_max_comments", 250)
            current_min_cluster_size = st.session_state.get("cluster_min_size", 3)
            
            cluster_button = st.button("Cluster Comments", 
                    type='primary', 
                    use_container_width=True, 
                    key="cluster_btn")
            
            # Cluster colors
            CLUSTER_COLORS = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
                "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700"
            ]
            
            if cluster_button:
                if not st.session_state.video_id:
                    st.error("Could not extract video ID from URL")
                else:
                    with st.spinner("Fetching comments and computing embeddings..."):
                        try:
                            # Fetch comments (with caching)
                            comments = get_cached_comments(st.session_state.video_id, max_comments=current_max_comments)
                            
                            if not comments:
                                st.warning("No comments found for this video.")
                            elif len(comments) < current_min_cluster_size:
                                st.warning(f"Not enough comments ({len(comments)}) for clustering. Need at least {current_min_cluster_size}.")
                            else:
                                st.success(f"Using {len(comments)} comments")
                                
                                # Clean texts
                                raw_texts = [c.get("text", "") for c in comments]
                                analysis_texts = [clean_text(t) for t in raw_texts]
                                display_texts = [clean_text_for_display(t) for t in raw_texts]
                                
                                # Filter out empty texts
                                valid_data = [(disp, anal) for disp, anal in zip(display_texts, analysis_texts) if anal.strip()]
                                
                                if valid_data and len(valid_data) >= current_min_cluster_size:
                                    display_texts, analysis_texts = zip(*valid_data)
                                    
                                    with st.spinner(f"Running HDBSCAN clustering (min_cluster_size={current_min_cluster_size})..."):
                                        cluster_result = cluster_comments(
                                            list(analysis_texts),
                                            min_cluster_size=current_min_cluster_size
                                        )
                                        
                                        # Pre-compute UMAP reduction to avoid recomputing on filter change
                                        reduced_embeddings = None
                                        if cluster_result["embeddings"] is not None and len(cluster_result["embeddings"]) > 5:
                                            with st.spinner("Computing visualization..."):
                                                try:
                                                    reduced_embeddings = reduce_dimensions_for_plot(cluster_result["embeddings"])
                                                except Exception:
                                                    reduced_embeddings = None
                                        
                                        # Extract top keywords per cluster
                                        cluster_keywords = {}
                                        if cluster_result["n_clusters"] > 0:
                                            with st.spinner("Extracting keywords..."):
                                                try:
                                                    cluster_keywords = top_keywords_per_cluster_nltk(
                                                        list(analysis_texts),
                                                        cluster_result["labels"],
                                                        top_n=10
                                                    )
                                                except Exception as kw_error:
                                                    st.warning(f"Could not extract keywords: {kw_error}")
                                                    cluster_keywords = {}
                                        
                                        st.session_state.cluster_data = {
                                            "display_texts": list(display_texts),
                                            "analysis_texts": list(analysis_texts),
                                            "labels": cluster_result["labels"],
                                            "embeddings": cluster_result["embeddings"],
                                            "reduced_embeddings": reduced_embeddings,
                                            "n_clusters": cluster_result["n_clusters"],
                                            "cluster_sizes": cluster_result["cluster_sizes"],
                                            "cluster_keywords": cluster_keywords,
                                            "settings": {
                                                "min_cluster_size": current_min_cluster_size,
                                                "max_comments": current_max_comments
                                            }
                                        }
                                        # Track URL to detect video changes
                                        st.session_state.previous_url = st.session_state[f"yt_{st.session_state.youtube_key}"]
                                else:
                                    st.warning("Not enough valid comments for clustering.")
                                    
                        except Exception as e:
                            st.error(f"Error during clustering: {str(e)}")
                            st.info("Make sure YOUTUBE_API_KEY is set in your environment variables.")
            
            # Display clustering results
            if st.session_state.cluster_data is not None:
                cluster_data = st.session_state.cluster_data
                n_clusters = cluster_data["n_clusters"]
                labels = cluster_data["labels"]
                cluster_sizes = cluster_data["cluster_sizes"]
                display_texts = cluster_data["display_texts"]
                embeddings = cluster_data["embeddings"]
                settings = cluster_data.get("settings", {})
                
                # Show settings used
                used_min_cluster = settings.get("min_cluster_size", "?")
                st.info(f"Clustered {len(display_texts)} comments with min_cluster_size={used_min_cluster}")
                
                if n_clusters == 0:
                    st.warning("No clusters found. Try adjusting the parameters or adding more comments.")
                else:
                    st.markdown(f"### Found {n_clusters} Clusters")
                    
                    # Cluster distribution
                    col_chart, col_table = st.columns([2, 1])
                    
                    with col_chart:
                        # Prepare data for pie chart (exclude noise cluster -1)
                        cluster_items = [(k, v) for k, v in cluster_sizes.items() if k >= 0]
                        cluster_items.sort(key=lambda x: x[1], reverse=True)
                        
                        pie_labels = [f"Cluster {k+1}" for k, _ in cluster_items]
                        pie_values = [v for _, v in cluster_items]
                        pie_colors = [CLUSTER_COLORS[k % len(CLUSTER_COLORS)] for k, _ in cluster_items]
                        
                        # Add noise if present
                        if -1 in cluster_sizes:
                            pie_labels.append("Noise")
                            pie_values.append(cluster_sizes[-1])
                            pie_colors.append("#666666")
                        
                        fig = px.pie(
                            values=pie_values,
                            names=pie_labels,
                            color_discrete_sequence=pie_colors,
                            hole=0.4
                        )
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent',
                            hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
                        )
                        fig.update_layout(
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=10)),
                            margin=dict(t=20, b=20, l=20, r=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_table:
                        table_items = [(k, v) for k, v in cluster_sizes.items()]
                        table_items.sort(key=lambda x: (x[0] == -1, -x[1]))  # Noise last, then by size
                        
                        total = len(labels)
                        table_data = {
                            "Cluster": [f"Cluster {k+1}" if k >= 0 else "Noise" for k, _ in table_items],
                            "#": [v for _, v in table_items],
                            "%": [f"{100*v/total:.1f}%" for _, v in table_items]
                        }
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    
                    # Top Keywords per Cluster
                    cluster_keywords = cluster_data.get("cluster_keywords", {})
                    if cluster_keywords:
                        st.markdown("### Top Keywords per Cluster")
                        
                        # Sort clusters by size (largest first)
                        sorted_clusters = sorted(
                            [(k, v) for k, v in cluster_sizes.items() if k >= 0],
                            key=lambda x: x[1], reverse=True
                        )
                        
                        for cluster_id, size in sorted_clusters:
                            keywords = cluster_keywords.get(cluster_id, [])
                            if keywords:
                                color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
                                
                                # Create keyword tags
                                keyword_tags = " ".join([
                                    f'<span style="background-color: {color}22; color: {color}; padding: 4px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 13px; border: 1px solid {color}44;">{kw}</span>'
                                    for kw in keywords[:8]  # Show top 8
                                ])
                                
                                st.markdown(f'''
                                    <div style="margin-bottom: 15px;">
                                        <p style="color: {color}; font-weight: bold; margin-bottom: 5px;">
                                            Cluster {cluster_id + 1} ({size} comments)
                                        </p>
                                        <div>{keyword_tags}</div>
                                    </div>
                                ''', unsafe_allow_html=True)
                    
                    # 2D visualization with UMAP (use cached reduced embeddings)
                    reduced = cluster_data.get("reduced_embeddings")
                    if reduced is not None and len(reduced) > 0:
                        st.markdown("### Cluster Visualization (UMAP)")
                        
                        # Create scatter plot
                        plot_df = pd.DataFrame({
                            'x': reduced[:, 0],
                            'y': reduced[:, 1],
                            'cluster': [f"Cluster {l+1}" if l >= 0 else "Noise" for l in labels],
                            'text': [t[:100] + "..." if len(t) > 100 else t for t in display_texts]
                        })
                        
                        fig_scatter = px.scatter(
                            plot_df, x='x', y='y', color='cluster',
                            hover_data={'text': True, 'x': False, 'y': False, 'cluster': True},
                            color_discrete_sequence=CLUSTER_COLORS[:n_clusters] + ["#666666"]
                        )
                        fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
                        fig_scatter.update_layout(
                            xaxis_title="", yaxis_title="",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(30,30,30,0.5)',
                            font=dict(color='white'),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True) 
        
        st.divider() 

        if st.button("Start again", type='secondary', icon=':material/refresh:'): 
            st.session_state.disabled_button = False
            st.session_state.youtube_key += 1
            st.session_state.condition_yt = False
            clear_outputs()
            st.rerun()
            
  
if __name__ == "__main__":
    main()
