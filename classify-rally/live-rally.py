# Live rally detector: continuously captures the screen with mss, runs every
# Nth frame through the rally classifier model, and displays a RALLY / NON-RALLY
# overlay in a preview window. Supports single-monitor (centered capture) and
# multi-monitor (preview shown on secondary display) setups. Captured frames
# are also saved to OUTPUT_DIR for later review.

import cv2
import numpy as np
from mss import mss
from dotenv import load_dotenv
from classify_rally.classify_rally import RallyClassifier
import time
import os
import uuid

load_dotenv()

MODEL_PATH = os.getenv("CLASSIFY_RALLY")
OUTPUT_DIR = "frames"
WINDOW_NAME = "Rally Classifier Live View"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_centered_screen_area(monitor, width=1280, height=720):
    left = monitor["left"] + (monitor["width"] - width) // 2
    top = monitor["top"] + (monitor["height"] - height) // 2
    return {"top": top, "left": left, "width": width, "height": height}


def get_top_screen_area(monitor, preview_reserved=260):
    """Capture only the TOP portion of the screen, leaving room for a preview window below."""
    capture_h = max(360, monitor["height"] - preview_reserved)
    capture_w = min(1280, monitor["width"] - 40)
    return {
        "top": monitor["top"],
        "left": monitor["left"] + (monitor["width"] - capture_w) // 2,
        "width": capture_w,
        "height": capture_h,
    }


def draw_overlay(frame, is_rally, confidence):
    color = (0, 255, 0) if is_rally else (0, 0, 255)
    label = f"STATUS: {'RALLY' if is_rally else 'NON-RALLY'}"
    conf_label = f"Confidence: {confidence*100:.1f}%"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, conf_label, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if is_rally:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)
    return frame


def run_single_monitor(classifier, sct, primary):
    """MacBook-only mode: centered capture + centered preview window (mirror effect will occur, user accepted)."""
    screen_area = get_centered_screen_area(primary, width=1280, height=720)

    print("--- Live Rally Classifier Active (Single Monitor Mode) ---")
    print("Watching live badminton — RALLY / NON-RALLY will appear in the preview window.")
    print("Note: a mirror/inception effect will appear in the preview, but the overlay stays readable.")
    print("Press 'q' in the preview window to stop.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_count = 0
    is_rally = False
    confidence = 0.0

    while True:
        sct_img = sct.grab(screen_area)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        if frame_count % 5 == 0:
            is_rally, confidence = classifier.predict(frame)
            status_text = "RALLY" if is_rally else "NON-RALLY"
            print(f"Frame {frame_count:05d}: {status_text} (Confidence: {confidence*100:.1f}%)")

            timestamp = int(time.time() * 1000)
            unique_id = str(uuid.uuid4())[:8]
            frame_filename = os.path.join(OUTPUT_DIR, f"live_{timestamp}_{unique_id}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame = draw_overlay(frame, is_rally, confidence)
        preview_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow(WINDOW_NAME, preview_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_multi_monitor(classifier, sct, screen_area, secondary_monitor):
    """Multi-monitor mode: preview window goes on the second monitor, safely outside the capture area."""
    print("--- Live Rally Classifier Active (Multi-Monitor Mode) ---")
    print(f"Preview will appear on secondary monitor at ({secondary_monitor['left']}, {secondary_monitor['top']}).")
    print("Press 'q' in the preview window to stop.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, secondary_monitor["left"] + 100, secondary_monitor["top"] + 100)

    frame_count = 0
    is_rally = False
    confidence = 0.0

    while True:
        sct_img = sct.grab(screen_area)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        if frame_count % 5 == 0:
            is_rally, confidence = classifier.predict(frame)
            status_text = "RALLY" if is_rally else "NON-RALLY"
            print(f"Frame {frame_count:05d}: {status_text} (Confidence: {confidence*100:.1f}%)")

            timestamp = int(time.time() * 1000)
            unique_id = str(uuid.uuid4())[:8]
            frame_filename = os.path.join(OUTPUT_DIR, f"live_{timestamp}_{unique_id}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame = draw_overlay(frame, is_rally, confidence)
        preview_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow(WINDOW_NAME, preview_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_live_classifier():
    classifier = RallyClassifier(MODEL_PATH, buffer_size=5)
    sct = mss()

    # mss.monitors[0] = union of all monitors, [1] = primary, [2+] = additional displays
    primary = sct.monitors[1]

    if len(sct.monitors) > 2:
        # Multi-monitor: capture is centered on primary, preview goes on secondary
        screen_area = get_centered_screen_area(primary, width=1280, height=720)
        run_multi_monitor(classifier, sct, screen_area, sct.monitors[2])
    else:
        # Single monitor: capture top, preview below
        run_single_monitor(classifier, sct, primary)


if __name__ == "__main__":
    run_live_classifier()
