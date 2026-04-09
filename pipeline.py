"""
TikTok Unified Analysis Pipeline

Fetches top posts from TikTok accounts, auto-detects post type
(slideshow vs video), and processes accordingly:
  - Slideshows: downloads slide images + metadata
  - Videos: downloads audio, transcribes with Whisper + metadata

Usage:
    python3 pipeline.py
    python3 pipeline.py --whisper-model medium --top-n 5
    python3 pipeline.py --force
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import fetch_top_posts
import analyzer_script
import video_transcript_script

# How many top posts (by views) to fetch per account when you run without --top-n.
TOP_POSTS_PER_ACCOUNT = 15


def check_dependencies():
    """Verify all required CLI tools and Python packages are installed."""
    missing = []
    for tool in ("gallery-dl", "yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        print(f"Missing required tools: {', '.join(missing)}")
        print("Install with: brew install ffmpeg && pip3 install gallery-dl yt-dlp")
        sys.exit(1)

    try:
        import whisper  # noqa: F401
    except ImportError:
        print("Missing required package: openai-whisper")
        print("Install with: pip3 install openai-whisper")
        sys.exit(1)


def process_post(url, whisper_model, data_dir="data", force=False):
    """
    Process a single TikTok post URL. Auto-detects type (slideshow vs video)
    and routes to the appropriate processing path.

    Returns "success", "skipped", or "failed".
    """
    project_root = Path(__file__).parent

    # Parse URL
    try:
        handle, post_id = analyzer_script.parse_url(url)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return "failed"

    # Check if already processed
    post_dir = Path(data_dir) / handle / post_id
    if not force and (post_dir / "post.json").exists():
        print(f"  Already processed: {post_dir}")
        return "skipped"

    post_dir.mkdir(parents=True, exist_ok=True)

    # Type detection: try gallery-dl for slides
    print(f"  Checking for slides ({handle}/{post_id})...")
    slides_ok = analyzer_script.download_slides(url, post_dir, project_root)
    slide_count, slide_filenames = analyzer_script.count_slides(post_dir)

    is_slideshow = slide_count > 0

    # Process based on detected type
    audio_ok = False
    transcript = None

    if is_slideshow:
        print(f"  Detected type: slideshow ({slide_count} slides)")
    else:
        print(f"  Detected type: video — downloading audio...")
        audio_ok = video_transcript_script.download_audio(url, post_dir)
        if not audio_ok:
            print(f"  WARNING: No audio downloaded")
        else:
            print(f"  Transcribing audio...")
            transcript = video_transcript_script.transcribe_audio(
                post_dir / "audio.mp3", whisper_model
            )
            if transcript:
                with open(post_dir / "transcript.txt", "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(f"  Transcript saved ({len(transcript)} chars)")
            else:
                print(f"  WARNING: Transcription failed")

            # Remove audio file — only the transcript is kept
            audio_path = post_dir / "audio.mp3"
            if audio_path.exists():
                audio_path.unlink()

    # Extract metadata (use video_transcript version — includes duration)
    print(f"  Extracting metadata...")
    metadata = video_transcript_script.extract_metadata(url)
    if not metadata:
        print(f"  WARNING: No metadata extracted, saving partial data")

    # Compute derived metrics
    derived = analyzer_script.compute_derived(metadata)

    # Format created_at
    created_at_ts = metadata.get("created_at", 0)
    if created_at_ts and isinstance(created_at_ts, (int, float)):
        created_at = datetime.fromtimestamp(
            created_at_ts, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        created_at = None

    # Assemble post data
    post_data = {
        "post_id": post_id,
        "handle": handle,
        "url": url,
        "type": "slideshow" if is_slideshow else "video",
        "caption": metadata.get("caption", ""),
        "created_at": created_at,
        "metrics": {
            "views": metadata.get("views", 0),
            "likes": metadata.get("likes", 0),
            "comments": metadata.get("comments", 0),
            "saves": metadata.get("saves", 0),
            "shares": metadata.get("shares", 0),
        },
        "derived": derived,
    }

    if is_slideshow:
        post_data["slide_count"] = slide_count
        post_data["slides"] = slide_filenames
    else:
        post_data["duration_seconds"] = metadata.get("duration", 0)
        post_data["transcript_file"] = "transcript.txt" if transcript else None

    # Save post.json
    output_path = post_dir / "post.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(post_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")

    if not is_slideshow and not audio_ok and not metadata:
        return "failed"
    return "success"


def main():
    parser = argparse.ArgumentParser(
        description="TikTok Unified Analysis Pipeline"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_POSTS_PER_ACCOUNT,
        help=f"Number of top posts per account (default: {TOP_POSTS_PER_ACCOUNT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process already downloaded posts",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Output data directory (default: data)",
    )
    parser.add_argument(
        "--accounts",
        default="accounts.txt",
        help="Path to accounts file (default: accounts.txt)",
    )
    args = parser.parse_args()

    # Check all dependencies
    check_dependencies()

    # Read accounts
    accounts_path = Path(args.accounts)
    if not accounts_path.exists():
        print(f"Error: {accounts_path} not found. Create it with one TikTok account URL per line.")
        sys.exit(1)

    accounts = [
        line.strip() for line in accounts_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not accounts:
        print("No accounts found in accounts file.")
        sys.exit(0)

    # Load Whisper model once
    whisper_model = video_transcript_script.load_whisper_model(args.whisper_model)

    print(f"\nFound {len(accounts)} accounts to process.\n")

    total_results = {"success": 0, "skipped": 0, "failed": 0}

    for account_url in accounts:
        # Extract handle for skip check
        match = re.search(r'tiktok\.com/@([\w._-]+)', account_url)
        if not match:
            print(f"Skipping invalid account URL: {account_url}")
            continue

        handle = f"@{match.group(1)}"

        print(f"{'='*60}")
        print(f"Processing Account: {account_url}")
        print(f"{'='*60}")

        try:
            # Fetch top posts
            post_urls = fetch_top_posts.fetch_top_urls(account_url, args.top_n)

            if not post_urls:
                print(f"  No posts found for {account_url}")
                continue

            print(f"\n  Found {len(post_urls)} top posts. Starting analysis...\n")

            # Process each post
            for i, post_url in enumerate(post_urls, 1):
                print(f"  [{i}/{len(post_urls)}] Processing: {post_url}")
                try:
                    status = process_post(
                        post_url, whisper_model,
                        data_dir=args.data_dir, force=args.force,
                    )
                    total_results[status] += 1
                except Exception as e:
                    print(f"  FAILED: {e}")
                    total_results["failed"] += 1

        except Exception as e:
            print(f"  Error processing account {account_url}: {e}")

    print(f"\n{'='*60}")
    print(f"Pipeline Complete.")
    print(f"Success: {total_results['success']}, "
          f"Skipped: {total_results['skipped']}, "
          f"Failed: {total_results['failed']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
