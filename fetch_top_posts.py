"""
Fetch top posts from a TikTok account by views and write URLs to urls.txt.

Configure ACCOUNT_URL and TOP_N below, then run:
    python3 fetch_top_posts.py
"""

import yt_dlp

# ── Configuration ──────────────────────────────────────────────
# (Moved to main block to support module import)
# ───────────────────────────────────────────────────────────────


def fetch_top_urls(account_url, top_n):
    print(f"Fetching posts from {account_url}...")

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        data = ydl.extract_info(account_url, download=False)

    entries = list(data.get("entries", []))
    print(f"Found {len(entries)} total posts")

    entries.sort(key=lambda x: x.get("view_count", 0) or 0, reverse=True)
    top = entries[:top_n]

    for e in top:
        print(f"  {e.get('view_count', 0):>10} views  {e.get('url', '')}")

    return [e["url"] for e in top if e.get("url")]


if __name__ == "__main__":
    # Default values for standalone execution
    DEFAULT_ACCOUNT_URL = "https://www.tiktok.com/@forzic.bluebro"
    DEFAULT_TOP_N = 10

    urls = fetch_top_urls(DEFAULT_ACCOUNT_URL, DEFAULT_TOP_N)

    if not urls:
        print("No URLs found.")
        raise SystemExit(1)

    with open("urls.txt", "w") as f:
        f.write("\n".join(urls) + "\n")

    print(f"\nWrote {len(urls)} URLs to urls.txt")
