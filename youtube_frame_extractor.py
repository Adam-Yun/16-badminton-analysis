import argparse
import os
import sys
from typing import Optional

import cv2
import yt_dlp
from tqdm import tqdm


TARGET_FPS = 1
OUTPUT_DIR = "frames"
DOWNLOAD_DIR = "downloads"
URLS_FILE = "urls.txt"


DOWNLOAD_STRATEGIES = [
    {"name": "ios",      "player_client": ["ios"]},
    {"name": "android",  "player_client": ["android"]},
    {"name": "web",      "player_client": ["web"]},
    {"name": "tv",       "player_client": ["tv"]},
]

COOKIE_BROWSERS = ["chrome", "safari", "firefox", "brave", "edge"]


def _build_ydl_opts(download_dir: str, player_client: list, cookies_browser: Optional[str]) -> dict:
    opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(download_dir, "%(id)s.%(ext)s"),
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


def download_video(url: str, download_dir: str) -> str:
    os.makedirs(download_dir, exist_ok=True)

    attempts = []
    for strat in DOWNLOAD_STRATEGIES:
        attempts.append((f"player_client={strat['name']}", strat["player_client"], None))
    for browser in COOKIE_BROWSERS:
        attempts.append((f"cookies={browser}+ios", ["ios"], browser))

    last_error = None
    for label, player_client, cookies_browser in attempts:
        print(f"\n--- Trying download strategy: {label} ---")
        try:
            opts = _build_ydl_opts(download_dir, player_client, cookies_browser)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)

            base, _ = os.path.splitext(video_path)
            mp4_path = base + ".mp4"
            if os.path.exists(mp4_path):
                return mp4_path
            if os.path.exists(video_path):
                return video_path
            raise RuntimeError("Download finished but no output file was found.")
        except Exception as e:
            last_error = e
            print(f"Strategy '{label}' failed: {e}")

    raise RuntimeError(f"All download strategies failed. Last error: {last_error}")


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


def extract_frames(video_path: str, video_output_dir: str, video_id: str, target_fps: float) -> int:
    os.makedirs(video_output_dir, exist_ok=True)
    print(f"DEBUG: Attempting to extract from {video_path}")
    print(f"DEBUG: Saving to {video_output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        raise RuntimeError("Could not read source FPS from the video.")

    skip_interval = max(1, round(source_fps / target_fps))

    print(f"Source FPS: {source_fps:.2f}")
    print(f"Target FPS: {target_fps}")
    print(f"Skip interval: every {skip_interval}th frame")
    print(f"Total source frames: {total_frames}")

    frame_index = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % skip_interval == 0:
                saved_count += 1
                # Include video_id in the filename for global uniqueness
                filename = os.path.join(video_output_dir, f"frame_{video_id}_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)

            frame_index += 1
            pbar.update(1)

    cap.release()
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos listed in a text file and extract frames at a target FPS."
    )
    parser.add_argument(
        "urls_file", nargs="?", default=URLS_FILE,
        help=f"Path to a .txt file containing one YouTube URL per line (default: {URLS_FILE})",
    )
    parser.add_argument(
        "--fps", type=float, default=TARGET_FPS,
        help=f"Target frames per second to extract (default: {TARGET_FPS})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Output directory for extracted frames (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--download-dir", default=DOWNLOAD_DIR,
        help=f"Directory to temporarily store the downloaded video (default: {DOWNLOAD_DIR})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.urls_file):
        print(f"URL file not found: {args.urls_file}")
        sys.exit(1)

    urls = read_urls(args.urls_file)
    if not urls:
        print(f"No URLs found in {args.urls_file}.")
        sys.exit(1)

    print(f"Found {len(urls)} URL(s) in {args.urls_file}.")

    total_saved = 0
    for i, url in enumerate(urls, start=1):
        print(f"\n=== [{i}/{len(urls)}] {url} ===")
        video_path = None
        try:
            print("Downloading video...")
            video_path = download_video(url, args.download_dir)
            print(f"Downloaded: {video_path}")

            video_id = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(args.output, video_id)

            saved = extract_frames(video_path, video_output_dir, video_id, args.fps)
            total_saved += saved
            print(f"Saved {saved} frames to '{video_output_dir}'.")
        except Exception as e:
            print(f"Failed to process {url}: {e}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted source video: {video_path}")

    print(f"\nDone. Total frames saved across all videos: {total_saved}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)


'''
https://www.youtube.com/watch?v=Wsv9c9iwxFY
'''