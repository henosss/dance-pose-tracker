#!/usr/bin/env python3
"""
AI Dance Teacher
================
Detects your dance moves in real-time using YOLOv8 pose estimation,
identifies the dance style, gives step-by-step instructions, and
prints the full move sequence at the end.

Usage
-----
  python main.py                        # webcam
  python main.py --source video.mp4     # local video
  python main.py --source "https://www.youtube.com/watch?v=..."  # YouTube
  python main.py --source 0 --mirror    # mirrored webcam (easier to follow)
"""

import argparse
import sys
import os
import cv2
import numpy as np

from pose_detector import PoseDetector
from dance_classifier import classify
from dance_teacher import DanceTeacher
from visualizer import draw_skeleton, draw_hud


def download_youtube(url: str) -> str:
    """Download a YouTube video and return the local file path."""
    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp not installed. Run: pip install yt-dlp")
        sys.exit(1)

    out_path = "yt_download.mp4"
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "quiet": True,
        "merge_output_format": "mp4",
    }
    print(f"Downloading: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete.\n")
    return out_path


def open_source(source: str):
    """Return an OpenCV VideoCapture and estimated fps."""
    if source.startswith("http"):
        path = download_youtube(source)
        cap = cv2.VideoCapture(path)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cap, fps


def run(source: str, mirror: bool, output: str | None, model: str, conf: float):
    cap, fps = open_source(source)
    detector = PoseDetector(model_path=model, conf=conf)
    teacher = DanceTeacher(history_seconds=3, fps=fps, stability_frames=max(4, int(fps / 5)))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
        print(f"Saving output to: {output}")

    print("AI Dance Teacher running — press Q to quit.\n")
    print("┌─────────────────────────────────────────────┐")
    print("│  Move in front of the camera and dance!     │")
    print("│  Your moves will appear on screen with      │")
    print("│  step-by-step teaching instructions.        │")
    print("└─────────────────────────────────────────────┘\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if mirror:
            frame = cv2.flip(frame, 1)

        people = detector.detect(frame)

        # Teach based on the first (most prominent) person detected
        if people:
            kp = people[0]
            move, score = classify(kp)
            teacher.update(move, score)
            draw_skeleton(frame, kp)
        else:
            teacher.update({"name": "No person", "style": "general",
                            "instruction": "Step into frame so the camera can see you.",
                            "next_hint": ""}, 0.0)

        draw_hud(frame, teacher)

        if writer:
            writer.write(frame)

        cv2.imshow("AI Dance Teacher", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    teacher.print_session_summary()


def main():
    parser = argparse.ArgumentParser(description="AI Dance Teacher — YOLOv8 Pose")
    parser.add_argument("--source", default="0",
                        help="Webcam index (0), local video path, or YouTube URL")
    parser.add_argument("--mirror", action="store_true",
                        help="Flip webcam horizontally (easier to follow along)")
    parser.add_argument("--output", default=None,
                        help="Save annotated video to this path (e.g. out.mp4)")
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="YOLOv8 pose model (yolov8n-pose.pt / yolov8s-pose.pt etc)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold (0-1)")
    args = parser.parse_args()
    run(args.source, args.mirror, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()
