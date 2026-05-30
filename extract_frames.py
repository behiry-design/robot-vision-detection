#!/usr/bin/env python3
"""
Video Frame Extraction Script
Extracts 1 frame per second from videos for dataset creation
"""
import cv2
import os

# ── Configuration ──────────────────────────────────────────────────────────
# Add your video paths here
VIDEOS = [
    ("path/to/your/video1.mp4", "prefix1"),
    ("path/to/your/video2.mp4", "prefix2"),
]

OUTPUT_DIR = "extracted_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Extract frames ──────────────────────────────────────────────────────────
total = 0
for video_path, prefix in VIDEOS:
    if not os.path.exists(video_path):
        print(f"NOT FOUND: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"{prefix}: FPS={fps:.1f} | Frames={total_frames} | Duration={duration:.1f}s")

    # Extract 1 frame per second
    frame_interval = max(1, int(fps))
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = f"{OUTPUT_DIR}/{prefix}_frame_{count:04d}.jpg"
            cv2.imwrite(out_path, frame)
            count += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted: {count} frames")
    total += count

print(f"\nTotal frames extracted: {total}")
print(f"Saved to: {OUTPUT_DIR}")
