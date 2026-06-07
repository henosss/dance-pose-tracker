#!/usr/bin/env python3
"""
Dance Pose Tracker — YOLOv9 powered pose detection and dance move classification.

Usage:
    python dance_pose_tracker.py --url <youtube_url> [options]
    python dance_pose_tracker.py --video <local_video_path> [options]
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


# ── Keypoint indices (COCO 17-point skeleton) ──────────────────────────────────
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

SKELETON = [
    (KP["nose"], KP["left_eye"]), (KP["nose"], KP["right_eye"]),
    (KP["left_eye"], KP["left_ear"]), (KP["right_eye"], KP["right_ear"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]), (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]), (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]), (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]), (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]), (KP["right_knee"], KP["right_ankle"]),
]

# Colours
SKEL_COLOR = (0, 255, 128)
KP_COLOR = (255, 80, 80)
TEXT_COLOR = (255, 255, 255)
OVERLAY_COLOR = (20, 20, 20)
MOVE_COLORS = {
    "Arms Raised": (255, 200, 0),
    "Jump / Airborne": (0, 220, 255),
    "Squat / Low": (180, 0, 255),
    "Spin / Twist": (255, 100, 0),
    "Side Stretch": (0, 255, 180),
    "Standing": (100, 255, 100),
    "Unknown": (180, 180, 180),
}


# ── Dance move classifier ──────────────────────────────────────────────────────

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by rays b→a and b→c (degrees)."""
    v1 = a - b
    v2 = c - b
    cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _kp(kps: np.ndarray, name: str) -> np.ndarray:
    """Return (x, y) for a named keypoint."""
    return kps[KP[name], :2]


def _visible(kps: np.ndarray, *names: str, conf_thresh: float = 0.3) -> bool:
    """True only when all named keypoints have confidence > threshold."""
    for n in names:
        if kps[KP[n], 2] < conf_thresh:
            return False
    return True


class DanceMoveClassifier:
    """Stateful classifier — uses a short history to smooth labels."""

    HISTORY = 8

    def __init__(self):
        self._history: list[str] = []
        self._prev_hip_y: float | None = None
        self._hip_y_buffer: list[float] = []

    def classify(self, kps: np.ndarray) -> str:
        """
        kps: (17, 3) array of (x, y, confidence) in pixel coords.
        Returns a human-readable dance move label.
        """
        move = self._raw_classify(kps)
        self._history.append(move)
        if len(self._history) > self.HISTORY:
            self._history.pop(0)
        # majority vote over history
        from collections import Counter
        return Counter(self._history).most_common(1)[0][0]

    def _raw_classify(self, kps: np.ndarray) -> str:
        # ── Arms raised ──────────────────────────────────────────────────────
        if _visible(kps, "left_wrist", "left_shoulder", "right_wrist", "right_shoulder"):
            lw = _kp(kps, "left_wrist")
            ls = _kp(kps, "left_shoulder")
            rw = _kp(kps, "right_wrist")
            rs = _kp(kps, "right_shoulder")
            # in image coords y increases downward — wrist above shoulder = smaller y
            if lw[1] < ls[1] - 20 and rw[1] < rs[1] - 20:
                return "Arms Raised"
            if lw[1] < ls[1] - 20 or rw[1] < rs[1] - 20:
                return "Arms Raised"

        # ── Jump / airborne — ankles high relative to hips ───────────────────
        if _visible(kps, "left_ankle", "right_ankle", "left_hip", "right_hip"):
            hip_y = (_kp(kps, "left_hip")[1] + _kp(kps, "right_hip")[1]) / 2
            ank_y = (_kp(kps, "left_ankle")[1] + _kp(kps, "right_ankle")[1]) / 2
            self._hip_y_buffer.append(hip_y)
            if len(self._hip_y_buffer) > 10:
                self._hip_y_buffer.pop(0)
            baseline_hip = np.median(self._hip_y_buffer)
            if hip_y < baseline_hip - 30:  # body elevated
                return "Jump / Airborne"

        # ── Squat / low ───────────────────────────────────────────────────────
        if _visible(kps, "left_knee", "right_knee", "left_hip", "right_hip",
                    "left_ankle", "right_ankle"):
            knee_y = (_kp(kps, "left_knee")[1] + _kp(kps, "right_knee")[1]) / 2
            hip_y = (_kp(kps, "left_hip")[1] + _kp(kps, "right_hip")[1]) / 2
            ank_y = (_kp(kps, "left_ankle")[1] + _kp(kps, "right_ankle")[1]) / 2
            leg_len = ank_y - hip_y + 1e-6
            squat_ratio = (knee_y - hip_y) / leg_len
            if squat_ratio > 0.55:
                return "Squat / Low"

        # ── Spin / twist — shoulder–hip misalignment ─────────────────────────
        if _visible(kps, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            ls = _kp(kps, "left_shoulder")
            rs = _kp(kps, "right_shoulder")
            lh = _kp(kps, "left_hip")
            rh = _kp(kps, "right_hip")
            shoulder_mid_x = (ls[0] + rs[0]) / 2
            hip_mid_x = (lh[0] + rh[0]) / 2
            shoulder_width = abs(rs[0] - ls[0]) + 1e-6
            twist = abs(shoulder_mid_x - hip_mid_x) / shoulder_width
            if twist > 0.35:
                return "Spin / Twist"

        # ── Side stretch — torso tilt ─────────────────────────────────────────
        if _visible(kps, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            ls = _kp(kps, "left_shoulder")
            rs = _kp(kps, "right_shoulder")
            shoulder_tilt = abs(ls[1] - rs[1]) / (abs(ls[0] - rs[0]) + 1e-6)
            if shoulder_tilt > 0.5:
                return "Side Stretch"

        if _visible(kps, "left_ankle", "right_ankle"):
            return "Standing"

        return "Unknown"


# ── Video download ─────────────────────────────────────────────────────────────

def download_youtube(url: str, output_dir: str = "downloads") -> str:
    """Download a YouTube video with yt-dlp and return the local file path."""
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # yt-dlp may change extension after merge
        if not os.path.exists(filename):
            filename = filename.rsplit(".", 1)[0] + ".mp4"
    return filename


# ── Rendering helpers ──────────────────────────────────────────────────────────

def draw_skeleton(frame: np.ndarray, kps: np.ndarray, conf_thresh: float = 0.3):
    """Draw skeleton lines and keypoint circles onto frame (in-place)."""
    h, w = frame.shape[:2]
    for i, j in SKELETON:
        if kps[i, 2] > conf_thresh and kps[j, 2] > conf_thresh:
            x1, y1 = int(kps[i, 0]), int(kps[i, 1])
            x2, y2 = int(kps[j, 0]), int(kps[j, 1])
            cv2.line(frame, (x1, y1), (x2, y2), SKEL_COLOR, 2, cv2.LINE_AA)

    for idx in range(17):
        if kps[idx, 2] > conf_thresh:
            cx, cy = int(kps[idx, 0]), int(kps[idx, 1])
            cv2.circle(frame, (cx, cy), 4, KP_COLOR, -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, move: str, person_id: int,
             fps: float, frame_no: int, total: int):
    """Draw a semi-transparent HUD overlay with current move label."""
    h, w = frame.shape[:2]
    color = MOVE_COLORS.get(move, MOVE_COLORS["Unknown"])

    # top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), OVERLAY_COLOR, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"Person {person_id}  |  {move}",
                (12, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

    # bottom bar — fps / progress
    cv2.rectangle(frame, (0, h - 30), (w, h), OVERLAY_COLOR, -1)
    progress = f"Frame {frame_no}/{total}  |  {fps:.1f} FPS  |  YOLOv9 Pose"
    cv2.putText(frame, progress, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_move_label(frame: np.ndarray, move: str, bbox):
    """Draw bounding-box and move label above the detected person."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = MOVE_COLORS.get(move, MOVE_COLORS["Unknown"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f" {move} "
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    lx, ly = x1, max(y1 - th - 8, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw, ly + th + 8), color, -1)
    cv2.putText(frame, label, (lx, ly + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


# ── Core processing ────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_path: str,
    model_name: str = "yolov9c-pose.pt",
    conf: float = 0.4,
    device: str = "",
    show: bool = False,
    max_frames: int = 0,
) -> dict:
    """
    Run YOLOv9 pose detection on every frame, classify dance moves, write output.

    Returns a summary dict with per-frame move labels.
    """
    print(f"\n[*] Loading model: {model_name}")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print(f"[*] Video: {width}x{height} @ {orig_fps:.1f} fps  ({total_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

    classifiers: dict[int, DanceMoveClassifier] = {}
    summary: list[dict] = []
    frame_no = 0
    t_start = time.time()

    with tqdm(total=total_frames, unit="frame", desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_no >= max_frames):
                break

            results = model.predict(
                frame,
                conf=conf,
                device=device,
                verbose=False,
                classes=[0],  # persons only
            )

            elapsed = time.time() - t_start + 1e-6
            fps = (frame_no + 1) / elapsed

            frame_moves: list[str] = []

            for result in results:
                if result.keypoints is None:
                    continue
                kps_data = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
                boxes = result.boxes.xyxy.cpu().numpy()           # (N, 4)

                for person_idx in range(len(kps_data)):
                    pid = person_idx  # simple id; replace with tracker id if available
                    kps = kps_data[person_idx]  # (17, 3)

                    if pid not in classifiers:
                        classifiers[pid] = DanceMoveClassifier()
                    move = classifiers[pid].classify(kps)
                    frame_moves.append(move)

                    draw_skeleton(frame, kps)
                    if person_idx < len(boxes):
                        draw_move_label(frame, move, boxes[person_idx])

            # HUD uses first detected person's move (or blank)
            primary_move = frame_moves[0] if frame_moves else "No person detected"
            draw_hud(frame, primary_move, 0, fps, frame_no + 1, total_frames)

            summary.append({"frame": frame_no, "move": primary_move, "fps": fps})
            writer.write(frame)

            if show:
                cv2.imshow("Dance Pose Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_no += 1
            pbar.update(1)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    print(f"\n[+] Done — {frame_no} frames in {elapsed_total:.1f}s "
          f"({frame_no / elapsed_total:.1f} fps avg)")
    print(f"[+] Output saved: {output_path}")
    return {"frames": frame_no, "summary": summary}


def print_stats(summary: list[dict]):
    """Print move frequency statistics from the run summary."""
    from collections import Counter
    moves = [s["move"] for s in summary]
    counts = Counter(moves)
    total = len(moves)
    print("\n── Dance Move Statistics ───────────────────────────────")
    for move, count in counts.most_common():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {move:<22} {bar:<50} {pct:5.1f}%  ({count} frames)")
    print("────────────────────────────────────────────────────────\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dance Pose Tracker — YOLOv9 powered",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL to download and process")
    src.add_argument("--video", help="Path to a local video file")

    p.add_argument("--output", "-o", default="output_dance.mp4",
                   help="Output annotated video path")
    p.add_argument("--model", default="yolov9c-pose.pt",
                   help="YOLOv9 pose model weights (auto-downloaded if missing)")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Detection confidence threshold")
    p.add_argument("--device", default="",
                   help="Inference device: '' (auto), 'cpu', '0', '0,1', ...")
    p.add_argument("--show", action="store_true",
                   help="Display live preview window while processing")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Limit processing to first N frames (0 = all)")
    p.add_argument("--download-dir", default="downloads",
                   help="Directory for downloaded YouTube videos")
    return p


def main():
    args = build_parser().parse_args()

    if args.url:
        print(f"[*] Downloading: {args.url}")
        video_path = download_youtube(args.url, args.download_dir)
        print(f"[*] Downloaded: {video_path}")
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"[!] File not found: {video_path}", file=sys.stderr)
            sys.exit(1)

    result = process_video(
        video_path=video_path,
        output_path=args.output,
        model_name=args.model,
        conf=args.conf,
        device=args.device,
        show=args.show,
        max_frames=args.max_frames,
    )
    print_stats(result["summary"])


if __name__ == "__main__":
    main()
