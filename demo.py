#!/usr/bin/env python3
"""
Quick demo — generates a synthetic video with a stick figure "dancing",
then runs the dance pose tracker on it so you can verify everything works
without needing a real YouTube URL.
"""

import cv2
import math
import numpy as np
import subprocess
import sys
import tempfile
import os

W, H = 640, 480
FPS = 30
DURATION = 5  # seconds


def synth_pose(t: float) -> list[tuple[int, int]]:
    """Return 17 (x, y) landmark coords that animate over time t (seconds)."""
    cx, cy = W // 2, H // 2

    # body sway
    sway = int(30 * math.sin(2 * math.pi * t * 0.8))

    # arm wave
    arm_angle = math.pi / 4 + math.pi / 4 * math.sin(2 * math.pi * t * 1.5)

    # leg bend (squat pulse)
    squat = int(20 * max(0, math.sin(2 * math.pi * t * 0.6)))

    nose        = (cx + sway,       cy - 130)
    left_eye    = (cx + sway - 12,  cy - 140)
    right_eye   = (cx + sway + 12,  cy - 140)
    left_ear    = (cx + sway - 20,  cy - 132)
    right_ear   = (cx + sway + 20,  cy - 132)
    left_sh     = (cx + sway - 40,  cy - 100)
    right_sh    = (cx + sway + 40,  cy - 100)

    lel_x = left_sh[0]  - int(50 * math.cos(arm_angle))
    lel_y = left_sh[1]  + int(50 * math.sin(arm_angle))
    lwrist_x = lel_x    - int(40 * math.cos(arm_angle))
    lwrist_y = lel_y    + int(40 * math.sin(arm_angle))

    rel_x = right_sh[0] + int(50 * math.cos(arm_angle))
    rel_y = right_sh[1] + int(50 * math.sin(arm_angle))
    rwrist_x = rel_x    + int(40 * math.cos(arm_angle))
    rwrist_y = rel_y    + int(40 * math.sin(arm_angle))

    left_hip  = (cx + sway - 30, cy - 20)
    right_hip = (cx + sway + 30, cy - 20)

    left_knee  = (cx + sway - 35, cy + 60 + squat)
    right_knee = (cx + sway + 35, cy + 60 + squat)
    left_ank   = (cx + sway - 35, cy + 130 + squat // 2)
    right_ank  = (cx + sway + 35, cy + 130 + squat // 2)

    return [
        nose, left_eye, right_eye, left_ear, right_ear,
        left_sh, right_sh,
        (lel_x, lel_y), (rel_x, rel_y),
        (lwrist_x, lwrist_y), (rwrist_x, rwrist_y),
        left_hip, right_hip,
        left_knee, right_knee,
        left_ank, right_ank,
    ]


SKEL = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_stick(frame, pts):
    for i, j in SKEL:
        cv2.line(frame, pts[i], pts[j], (0, 220, 100), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 5, (255, 80, 80), -1, cv2.LINE_AA)


def create_synth_video(path: str):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = DURATION * FPS
    for f in range(frames):
        t = f / FPS
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        # gradient background
        canvas[:, :] = (30, 20, 60)
        pts = synth_pose(t)
        draw_stick(canvas, pts)
        cv2.putText(canvas, "Synthetic Dance Demo", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        out.write(canvas)
    out.release()
    print(f"[demo] Synthetic video written: {path}")


if __name__ == "__main__":
    tmp = tempfile.mktemp(suffix=".mp4")
    create_synth_video(tmp)

    out = "demo_output.mp4"
    cmd = [
        sys.executable, "dance_pose_tracker.py",
        "--video", tmp,
        "--output", out,
        "--conf", "0.2",
        "--max-frames", "60",
    ]
    print(f"[demo] Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    os.unlink(tmp)
    print(f"\n[demo] Output: {out}")
