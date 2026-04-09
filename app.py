"""
TikTok Analyzer — Streamlit Web App

Wraps the TikTok analysis pipeline in a simple UI.
Users enter account URLs, choose how many top posts to analyze,
and get a sortable results table with metrics.
"""

import re
import streamlit as st
import pandas as pd
from pathlib import Path

import fetch_top_posts
import analyzer_script
import video_transcript_script
import pipeline
from utils import capture_stdout, check_deps, load_results, format_number

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cached Whisper model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def get_whisper_model(model_name: str):
    """Load Whisper model once and cache across reruns."""
    return video_transcript_script.load_whisper_model(model_name)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TikTok Analyzer", page_icon="📊", layout="wide")
st.title("TikTok Post Analyzer")
st.caption("Analyze top-performing TikTok posts by account")

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
missing = check_deps()
if missing:
    st.error(
        f"Missing dependencies: **{', '.join(missing)}**\n\n"
        "Install with: `pip install gallery-dl yt-dlp openai-whisper` and `brew install ffmpeg`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    account_urls_text = st.text_area(
        "TikTok Account URLs",
        placeholder="https://www.tiktok.com/@username\nhttps://www.tiktok.com/@another",
        help="One account URL per line",
        height=150,
    )

    top_n = st.slider("Top posts per account", min_value=1, max_value=50, value=10)

    whisper_model_name = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=0,
        help="Tiny is fastest. Small is most accurate but uses more memory.",
    )

    force = st.checkbox("Force re-process", value=False, help="Re-analyze posts that were already processed")

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Parse and validate account URLs
# ---------------------------------------------------------------------------
def parse_accounts(text: str) -> list[str]:
    """Parse text area into list of valid TikTok account URLs."""
    urls = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.search(r'tiktok\.com/@[\w._-]+', line):
            urls.append(line)
    return urls


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if run_button:
    accounts = parse_accounts(account_urls_text)

    if not accounts:
        st.error("Please enter at least one valid TikTok account URL (e.g. https://www.tiktok.com/@username)")
        st.stop()

    # Load Whisper model
    with st.spinner(f"Loading Whisper model ({whisper_model_name})..."):
        whisper_model = get_whisper_model(whisper_model_name)

    # Track which handles we process
    processed_handles = []
    total_results = {"success": 0, "skipped": 0, "failed": 0}

    with st.status("Analyzing TikTok accounts...", expanded=True) as status:
        for acc_idx, account_url in enumerate(accounts):
            # Extract handle
            match = re.search(r'tiktok\.com/@([\w._-]+)', account_url)
            if not match:
                st.write(f"Skipping invalid URL: {account_url}")
                continue

            handle = f"@{match.group(1)}"
            processed_handles.append(handle)

            st.write(f"**[{acc_idx + 1}/{len(accounts)}] Fetching top posts for {handle}...**")

            # Fetch top post URLs
            try:
                with capture_stdout() as buf:
                    post_urls = fetch_top_posts.fetch_top_urls(account_url, top_n)
            except Exception as e:
                st.write(f"Error fetching posts for {handle}: {e}")
                continue

            if not post_urls:
                st.write(f"No posts found for {handle}")
                continue

            st.write(f"Found {len(post_urls)} posts. Processing...")

            # Progress bar for this account
            progress = st.progress(0)

            for i, post_url in enumerate(post_urls):
                st.write(f"  Processing post {i + 1}/{len(post_urls)}...")
                try:
                    with capture_stdout() as buf:
                        result = pipeline.process_post(
                            post_url,
                            whisper_model,
                            data_dir=str(DATA_DIR),
                            force=force,
                        )
                    total_results[result] += 1
                except Exception as e:
                    st.write(f"  Failed: {e}")
                    total_results["failed"] += 1

                progress.progress((i + 1) / len(post_urls))

        status.update(
            label=f"Done! {total_results['success']} processed, "
                  f"{total_results['skipped']} skipped, "
                  f"{total_results['failed']} failed",
            state="complete",
        )

    # Store results in session state so they persist
    st.session_state["processed_handles"] = processed_handles

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
handles_to_show = st.session_state.get("processed_handles", [])

# Also allow browsing previously analyzed data
if DATA_DIR.exists():
    existing_handles = sorted(
        d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("@")
    )
    if existing_handles and not handles_to_show:
        handles_to_show = existing_handles

if handles_to_show:
    results = load_results(DATA_DIR, handles_to_show)

    if results:
        st.divider()
        st.subheader(f"Results ({len(results)} posts)")

        # Build DataFrame
        rows = []
        for post in results:
            m = post.get("metrics", {})
            d = post.get("derived", {})
            rows.append({
                "Handle": post.get("handle", ""),
                "Type": post.get("type", "unknown"),
                "Views": m.get("views", 0),
                "Likes": m.get("likes", 0),
                "Comments": m.get("comments", 0),
                "Saves": m.get("saves", 0),
                "Shares": m.get("shares", 0),
                "Engagement": d.get("engagement_rate", 0),
                "Age (days)": d.get("age_days", 0),
                "Caption": post.get("caption", "")[:100],
                "URL": post.get("url", ""),
            })

        df = pd.DataFrame(rows)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Posts", len(df))
        col2.metric("Total Views", format_number(df["Views"].sum()))
        col3.metric("Avg Engagement", f"{df['Engagement'].mean():.1%}")
        col4.metric("Accounts", df["Handle"].nunique())

        # Main data table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Views": st.column_config.NumberColumn(format="%d"),
                "Likes": st.column_config.NumberColumn(format="%d"),
                "Comments": st.column_config.NumberColumn(format="%d"),
                "Saves": st.column_config.NumberColumn(format="%d"),
                "Shares": st.column_config.NumberColumn(format="%d"),
                "Engagement": st.column_config.NumberColumn(format="%.1%%"),
                "URL": st.column_config.LinkColumn("URL"),
            },
        )

        # Per-account detail sections
        st.divider()
        st.subheader("Post Details")

        for handle in sorted(set(r.get("handle", "") for r in results)):
            handle_posts = [r for r in results if r.get("handle") == handle]
            with st.expander(f"{handle} ({len(handle_posts)} posts)"):
                for post in handle_posts:
                    m = post.get("metrics", {})
                    post_type = post.get("type", "unknown")

                    st.markdown(f"**{post_type.title()}** — {format_number(m.get('views', 0))} views | "
                                f"{format_number(m.get('likes', 0))} likes | "
                                f"{format_number(m.get('saves', 0))} saves")

                    st.markdown(f"*{post.get('caption', 'No caption')[:200]}*")

                    # Show slides for slideshow posts
                    if post_type == "slideshow":
                        slides = post.get("slides", [])
                        post_dir = DATA_DIR / post.get("handle", "") / post.get("post_id", "")
                        slide_paths = [post_dir / s for s in slides if (post_dir / s).exists()]
                        if slide_paths:
                            cols = st.columns(min(len(slide_paths), 5))
                            for idx, slide_path in enumerate(slide_paths[:5]):
                                cols[idx].image(str(slide_path), use_container_width=True)
                            if len(slide_paths) > 5:
                                st.caption(f"+ {len(slide_paths) - 5} more slides")

                    # Show transcript for video posts
                    if post_type == "video":
                        transcript_file = post.get("transcript_file")
                        if transcript_file:
                            transcript_path = DATA_DIR / post.get("handle", "") / post.get("post_id", "") / transcript_file
                            if transcript_path.exists():
                                transcript = transcript_path.read_text()
                                st.text_area("Transcript", transcript, height=100, disabled=True,
                                             key=f"transcript_{post.get('post_id')}")

                    st.markdown(f"[Open on TikTok]({post.get('url', '')})")
                    st.divider()
else:
    st.info("Enter TikTok account URLs in the sidebar and click **Run Analysis** to get started.")
