"""
TikTok Video Transcript Extractor

Downloads audio and extracts transcripts from TikTok video posts
using yt-dlp and OpenAI Whisper.

Usage:
    python3 video_transcript_script.py
    python3 video_transcript_script.py --urls custom_urls.txt
    python3 video_transcript_script.py --whisper-model medium
    python3 video_transcript_script.py --force

Prerequisites:
    - ffmpeg: brew install ffmpeg
    - openai-whisper: pip install openai-whisper
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def check_dependencies():
    """Verify yt-dlp, ffmpeg, and whisper are installed."""
    missing = []
    for tool in ("yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        print(f"Missing required tools: {', '.join(missing)}")
        print("Install with: brew install ffmpeg && pip3 install yt-dlp")
        sys.exit(1)

    try:
        import whisper  # noqa: F401
    except ImportError:
        print("Missing required package: openai-whisper")
        print("Install with: pip3 install openai-whisper")
        sys.exit(1)


def parse_url(url):
    """
    Extract handle and post_id from a TikTok URL.

    Supports:
      https://www.tiktok.com/@user/photo/123456
      https://www.tiktok.com/@user/video/123456

    Returns (handle, post_id) e.g. ("@username", "123456")
    """
    pattern = r'tiktok\.com/@([\w._-]+)/(?:photo|video)/(\d+)'
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Cannot parse TikTok URL: {url}")
    handle = f"@{match.group(1)}"
    post_id = match.group(2)
    return handle, post_id


def download_audio(url, post_dir):
    """
    Download audio from a TikTok video using yt-dlp.

    Extracts audio as MP3 to post_dir/audio.mp3.
    Returns True if audio file was downloaded successfully.
    """
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", str(post_dir / "audio.%(ext)s"),
        url,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )
        if result.returncode != 0:
            print(f"  yt-dlp audio warning (exit {result.returncode}): {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("  yt-dlp audio download timed out after 180s")
        return False
    except FileNotFoundError:
        print("  yt-dlp not found")
        return False

    return (post_dir / "audio.mp3").exists()


def extract_metadata(url):
    """
    Extract metadata using yt-dlp --dump-json.

    Returns dict with: caption, views, likes, comments, shares, created_at, duration
    Returns empty dict if extraction fails.
    """
    normalized_url = url.replace("/photo/", "/video/")

    cmd = ["yt-dlp", "--dump-json", "--no-download", normalized_url]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  yt-dlp failed: {result.stderr[:200]}")
            return {}

        raw = json.loads(result.stdout)

        return {
            "caption": raw.get("description", ""),
            "views": raw.get("view_count", 0),
            "likes": raw.get("like_count", 0),
            "comments": raw.get("comment_count", 0),
            "saves": raw.get("save_count", 0),
            "shares": raw.get("repost_count", 0),
            "created_at": raw.get("timestamp", 0),
            "duration": raw.get("duration", 0),
        }
    except subprocess.TimeoutExpired:
        print("  yt-dlp timed out after 60s")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Failed to parse yt-dlp output: {e}")
        return {}
    except FileNotFoundError:
        print("  yt-dlp not found")
        return {}


def compute_derived(metadata):
    """
    Compute derived virality metrics.

    Returns dict with: age_days, engagement_rate
    """
    views = metadata.get("views", 0)
    likes = metadata.get("likes", 0)
    created_at = metadata.get("created_at", 0)

    age_days = 0
    if created_at:
        try:
            created_dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            age_days = max(1, (now - created_dt).days)
        except (ValueError, OSError):
            age_days = 0

    engagement_rate = round(likes / views, 3) if views > 0 else 0

    return {
        "age_days": age_days,
        "engagement_rate": engagement_rate,
    }


def load_whisper_model(model_name):
    """Load and return a Whisper model."""
    import whisper
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Whisper model loaded.")
    return model


def transcribe_audio(audio_path, model):
    """
    Transcribe an audio file using a pre-loaded Whisper model.

    Returns transcript text, or None if transcription fails.
    """
    try:
        result = model.transcribe(str(audio_path))
        return result.get("text", "").strip()
    except Exception as e:
        print(f"  Whisper transcription failed: {e}")
        return None


def process_url(url, model, data_dir="data", force=False):
    """
    Process a single TikTok video URL end-to-end.

    Returns "success", "skipped", or "failed".
    """
    # Parse URL
    try:
        handle, post_id = parse_url(url)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return "failed"

    # Check if already processed
    post_dir = Path(data_dir) / handle / post_id
    if not force and (post_dir / "post.json").exists():
        print(f"  Already processed: {post_dir}")
        return "skipped"

    post_dir.mkdir(parents=True, exist_ok=True)

    # Download audio
    print(f"  Downloading audio for {handle}/{post_id}...")
    audio_ok = download_audio(url, post_dir)
    if not audio_ok:
        print(f"  WARNING: No audio downloaded")

    # Extract metadata
    print(f"  Extracting metadata...")
    metadata = extract_metadata(url)
    if not metadata:
        print(f"  WARNING: No metadata extracted, saving partial data")

    # Transcribe audio
    transcript = None
    if audio_ok:
        print(f"  Transcribing audio...")
        transcript = transcribe_audio(post_dir / "audio.mp3", model)
        if transcript:
            with open(post_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"  Transcript saved ({len(transcript)} chars)")
        else:
            print(f"  WARNING: Transcription failed")

    # Compute derived metrics
    derived = compute_derived(metadata)

    # Format created_at
    created_at_ts = metadata.get("created_at", 0)
    if created_at_ts and isinstance(created_at_ts, (int, float)):
        created_at = datetime.fromtimestamp(created_at_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        created_at = None

    # Assemble and save post.json
    post_data = {
        "post_id": post_id,
        "handle": handle,
        "url": url,
        "type": "video",
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
        "duration_seconds": metadata.get("duration", 0),
        "audio_file": "audio.mp3" if audio_ok else None,
        "transcript_file": "transcript.txt" if transcript else None,
        "transcript": transcript,
    }

    output_path = post_dir / "post.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(post_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")

    if not audio_ok and not metadata:
        return "failed"
    return "success"


def main():
    parser = argparse.ArgumentParser(
        description="TikTok Video Transcript Extractor"
    )
    parser.add_argument("--urls", default="video_urls.txt", help="Path to URLs file")
    parser.add_argument("--force", action="store_true", help="Re-process already downloaded posts")
    parser.add_argument("--data-dir", default="data", help="Output data directory")
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (default: base)",
    )
    args = parser.parse_args()

    check_dependencies()

    urls_path = Path(args.urls)
    if not urls_path.exists():
        print(f"Error: {urls_path} not found. Create it with one TikTok video URL per line.")
        sys.exit(1)

    urls = [
        line.strip() for line in urls_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        print("No URLs found in file.")
        sys.exit(0)

    # Load Whisper model once
    model = load_whisper_model(args.whisper_model)

    results = {"success": 0, "skipped": 0, "failed": 0}
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        try:
            status = process_url(url, model, data_dir=args.data_dir, force=args.force)
            results[status] += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            results["failed"] += 1

    print(f"\nDone. Success: {results['success']}, "
          f"Skipped: {results['skipped']}, Failed: {results['failed']}")


if __name__ == "__main__":
    main()
