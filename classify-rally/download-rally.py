# Downloads YouTube videos listed in a URLs file using yt-dlp, saving them as
# merged mp4s in OUTPUT_DIR. Tries multiple player clients (ios/android/web/tv)
# and falls back to browser cookies, and skips URLs whose video ID is already
# present on disk.

import argparse
import os
import sys
from typing import Optional
from dotenv import load_dotenv
import yt_dlp

load_dotenv()

OUTPUT_DIR = "badminton_analysis_videos"
URLS_FILE = os.getenv("DOWNLOAD_VIDEO_URLS")

DOWNLOAD_STRATEGIES = [
    {"name": "web",     "player_client": ["web"]},
    {"name": "tv",      "player_client": ["tv"]},
    {"name": "ios",     "player_client": ["ios"]},
    {"name": "android", "player_client": ["android"]},
]

COOKIE_BROWSERS = ["chrome", "safari"]


def _build_ydl_opts(output_dir: str, player_client: list, cookies_browser: Optional[str]) -> dict:
    opts = {
        "format": "best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "extractor_args": {"youtube": {"player_client": player_client}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    return opts


def download_video(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    attempts = []
    for strat in DOWNLOAD_STRATEGIES:
        attempts.append((f"player_client={strat['name']}", strat["player_client"], None))
    for browser in COOKIE_BROWSERS:
        attempts.append((f"cookies={browser}+web", ["web"], browser))
        attempts.append((f"cookies={browser}+ios", ["ios"], browser))

    last_error = None
    for label, player_client, cookies_browser in attempts:
        print(f"  Trying strategy: {label}")
        try:
            opts = _build_ydl_opts(output_dir, player_client, cookies_browser)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)

            base, _ = os.path.splitext(video_path)
            mp4_path = base + ".mp4"
            if os.path.exists(mp4_path):
                return mp4_path
            if os.path.exists(video_path):
                return video_path
            raise RuntimeError("Download finished but output file not found.")
        except Exception as e:
            last_error = e
            print(f"  Strategy '{label}' failed: {e}")

    raise RuntimeError(f"All strategies failed. Last error: {last_error}")


def already_downloaded(url: str, output_dir: str) -> Optional[str]:
    """Return the existing file path if this URL was already downloaded."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id", "")
        if not video_id:
            return None
        for ext in ("mp4", "mkv", "webm"):
            candidate = os.path.join(output_dir, f"{video_id}.{ext}")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    return None


def read_urls(urls_file: str) -> list:
    with open(urls_file, "r") as f:
        lines = f.readlines()
    urls = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download badminton videos from a URL list into a folder."
    )
    parser.add_argument(
        "urls_file", nargs="?", default=URLS_FILE,
        help=f"Path to .txt file with one URL per line (default: {URLS_FILE})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Folder to save downloaded videos (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip videos that are already downloaded (default: on)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.urls_file):
        print(f"Error: URL file not found: {args.urls_file}")
        sys.exit(1)

    urls = read_urls(args.urls_file)
    if not urls:
        print(f"No URLs found in {args.urls_file}.")
        sys.exit(1)

    print(f"Found {len(urls)} URL(s). Saving to '{args.output}'.\n")

    ok, skipped, failed = 0, 0, 0

    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")

        if args.skip_existing:
            existing = already_downloaded(url, args.output)
            if existing:
                print(f"  Already downloaded: {existing} — skipping.\n")
                skipped += 1
                continue

        try:
            path = download_video(url, args.output)
            print(f"  Saved: {path}\n")
            ok += 1
        except Exception as e:
            print(f"  Failed: {e}\n")
            failed += 1

    print(f"Done. Downloaded: {ok}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
