"""
TikTok Slideshow Research Extractor

Downloads slide images and extracts metadata from TikTok slideshow posts.
Outputs structured JSON per post for later LLM analysis.

Usage:
    python3 analyzer_script.py
    python3 analyzer_script.py --urls custom_urls.txt
    python3 analyzer_script.py --force
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def check_dependencies():
    """Verify gallery-dl and yt-dlp are installed."""
    missing = []
    for tool in ("gallery-dl", "yt-dlp"):
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        print(f"Missing required tools: {', '.join(missing)}")
        print(f"Install with: pip3 install {' '.join(missing)}")
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


def download_slides(url, post_dir, project_root):
    """
    Download slideshow images using gallery-dl.

    Downloads to a temp subdirectory, then renames images to
    slide_01.jpg, slide_02.jpg, etc. in post_dir.

    Returns True if at least one image was downloaded.
    """
    config_path = project_root / "gallery-dl.conf"
    temp_dl_dir = post_dir / "_gallery_dl_temp"
    temp_dl_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gallery-dl",
        "--dest", str(temp_dl_dir),
        url,
    ]
    if config_path.exists():
        cmd.insert(1, "--config")
        cmd.insert(2, str(config_path))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  gallery-dl warning (exit {result.returncode}): {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("  gallery-dl timed out after 120s")
    except FileNotFoundError:
        print("  gallery-dl not found")
        _rmtree_safe(temp_dl_dir)
        return False

    # Find all downloaded images regardless of nested directory structure
    downloaded_images = sorted(
        f for f in temp_dl_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not downloaded_images:
        _rmtree_safe(temp_dl_dir)
        return False

    # Rename and move to post_dir as slide_01.ext, slide_02.ext, ...
    post_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(downloaded_images, 1):
        ext = img_path.suffix.lower()
        dest = post_dir / f"slide_{i:02d}{ext}"
        img_path.rename(dest)

    _rmtree_safe(temp_dl_dir)
    return True


def _rmtree_safe(path):
    """Safely remove a directory tree."""
    try:
        shutil.rmtree(path)
    except Exception:
        pass


def extract_metadata(url):
    """
    Extract metadata using yt-dlp --dump-json.

    Returns dict with: caption, views, likes, comments, shares, created_at
    Returns empty dict if extraction fails.
    """
    # yt-dlp may need /video/ instead of /photo/
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

    Returns dict with: age_days, views_per_day, engagement_rate, share_rate
    """
    views = metadata.get("views", 0)
    likes = metadata.get("likes", 0)
    comments = metadata.get("comments", 0)
    saves = metadata.get("saves", 0)
    shares = metadata.get("shares", 0)
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


def count_slides(post_dir):
    """
    Count image files in post_dir and return ordered filenames.

    Returns (slide_count, slide_filenames)
    """
    slides = sorted(
        f.name for f in post_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    return len(slides), slides


def process_url(url, data_dir="data", force=False):
    """
    Process a single TikTok slideshow URL end-to-end.

    Returns "success", "skipped", or "failed".
    """
    project_root = Path(__file__).parent

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

    # Download slides
    print(f"  Downloading slides for {handle}/{post_id}...")
    slides_ok = download_slides(url, post_dir, project_root)
    if not slides_ok:
        print(f"  WARNING: No slides downloaded")

    # Count slides
    slide_count, slide_filenames = count_slides(post_dir)
    print(f"  Found {slide_count} slides")

    # Extract metadata
    print(f"  Extracting metadata...")
    metadata = extract_metadata(url)
    if not metadata:
        print(f"  WARNING: No metadata extracted, saving partial data")

    # Compute derived metrics
    derived = compute_derived(metadata)

    # Assemble and save post.json
    created_at_ts = metadata.get("created_at", 0)
    if created_at_ts and isinstance(created_at_ts, (int, float)):
        created_at = datetime.fromtimestamp(created_at_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        created_at = None

    post_data = {
        "post_id": post_id,
        "handle": handle,
        "url": url,
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
        "slide_count": slide_count,
        "slides": slide_filenames,
    }

    output_path = post_dir / "post.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(post_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")

    if slide_count == 0 and not metadata:
        return "failed"
    return "success"


def main():
    parser = argparse.ArgumentParser(
        description="TikTok Slideshow Research Extractor"
    )
    parser.add_argument("--urls", default="urls.txt", help="Path to URLs file")
    parser.add_argument("--force", action="store_true", help="Re-process already downloaded posts")
    parser.add_argument("--data-dir", default="data", help="Output data directory")
    args = parser.parse_args()

    check_dependencies()

    urls_path = Path(args.urls)
    if not urls_path.exists():
        print(f"Error: {urls_path} not found. Create it with one TikTok URL per line.")
        sys.exit(1)

    urls = [
        line.strip() for line in urls_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        print("No URLs found in urls.txt")
        sys.exit(0)

    results = {"success": 0, "skipped": 0, "failed": 0}
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        try:
            status = process_url(url, data_dir=args.data_dir, force=args.force)
            results[status] += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            results["failed"] += 1

    print(f"\nDone. Success: {results['success']}, "
          f"Skipped: {results['skipped']}, Failed: {results['failed']}")


if __name__ == "__main__":
    main()
