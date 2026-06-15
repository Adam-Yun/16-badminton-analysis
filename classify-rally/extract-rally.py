# Extracts frames from local video files in VIDEOS_DIR at a target FPS using
# OpenCV, writing JPEGs named with the video ID into per-video subfolders under
# OUTPUT_DIR. Source videos are preserved.

import argparse
import os
import sys

import cv2
from tqdm import tqdm


TARGET_FPS = 1
VIDEOS_DIR = "/Users/adam/Desktop/Personal/Personal (Adam)/Github/16-badminton-analysis/videos"
OUTPUT_DIR = "frames"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def find_videos(videos_dir: str) -> list:
    if not os.path.isdir(videos_dir):
        return []
    files = []
    for name in sorted(os.listdir(videos_dir)):
        path = os.path.join(videos_dir, name)
        if os.path.isfile(path) and name.lower().endswith(VIDEO_EXTS):
            files.append(path)
    return files


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
                filename = os.path.join(video_output_dir, f"frame_{video_id}_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)

            frame_index += 1
            pbar.update(1)

    cap.release()
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from local videos at a target FPS."
    )
    parser.add_argument(
        "--videos-dir", default=VIDEOS_DIR,
        help=f"Folder containing local video files (default: {VIDEOS_DIR})",
    )
    parser.add_argument(
        "--fps", type=float, default=TARGET_FPS,
        help=f"Target frames per second to extract (default: {TARGET_FPS})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Output directory for extracted frames (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.videos_dir):
        print(f"Videos folder not found: {args.videos_dir}")
        sys.exit(1)

    videos = find_videos(args.videos_dir)
    if not videos:
        print(f"No video files found in {args.videos_dir}.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {args.videos_dir}.")

    total_saved = 0
    for i, video_path in enumerate(videos, start=1):
        print(f"\n=== [{i}/{len(videos)}] {video_path} ===")
        try:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(args.output, video_id)

            saved = extract_frames(video_path, video_output_dir, video_id, args.fps)
            total_saved += saved
            print(f"Saved {saved} frames to '{video_output_dir}'.")
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")

    print(f"\nDone. Total frames saved across all videos: {total_saved}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
