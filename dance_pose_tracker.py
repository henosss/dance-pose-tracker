"""High-performance distance-running gait & biomechanics analyzer.

Runs YOLO pose estimation frame-by-frame over a running video and extracts
biomechanical efficiency metrics from the detected skeleton:

  1. Knee flexion / extension (Hip-Knee-Ankle angle via Law of Cosines)
  2. Vertical oscillation of the hip centroid (rolling-variance baseline)
  3. Pelvic & torso stability (shoulder-midline vs hip-midline angular offset)
  4. Left vs. right asymmetry (running log of peak joint extensions)

Active joint angles and metrics are overlaid onto the runner's skeleton, and a
biomechanical efficiency breakdown is printed to the console on completion.

Usage:
    python dance_pose_tracker.py --source run.mp4 --output annotated.mp4
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "ultralytics is required. Install with: pip install ultralytics opencv-python"
    ) from exc


# Default to an official, cleanly-downloadable pose checkpoint.
# yolo26n-pose.pt is preferred; yolo11n-pose.pt is a stable fallback.
DEFAULT_MODEL = "yolo26n-pose.pt"
FALLBACK_MODEL = "yolo11n-pose.pt"

# COCO-17 keypoint indices used by YOLO *-pose models.
KP = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

# Minimum keypoint confidence to trust a joint for geometry.
KP_CONF = 0.4


def angle_law_of_cosines(a, b, c) -> float:
    """Interior angle (degrees) at vertex ``b`` for triangle a-b-c.

    Uses the Law of Cosines on the three side lengths so the result is
    rotation/translation invariant. ``a``, ``b``, ``c`` are (x, y) points.
    """
    ab = math.dist(a, b)
    bc = math.dist(b, c)
    ac = math.dist(a, c)
    if ab == 0 or bc == 0:
        return float("nan")
    # cos(B) = (ab^2 + bc^2 - ac^2) / (2 * ab * bc)
    cos_b = (ab * ab + bc * bc - ac * ac) / (2.0 * ab * bc)
    cos_b = max(-1.0, min(1.0, cos_b))  # clamp for numerical safety
    return math.degrees(math.acos(cos_b))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def line_angle_deg(p1, p2) -> float:
    """Angle of the line p1->p2 relative to the horizontal, in degrees."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


@dataclass
class SideStats:
    """Running peak-extension log for one leg."""

    name: str
    peak_extension: float = 0.0  # largest knee angle seen (straightest leg)
    min_flexion: float = 180.0   # smallest knee angle seen (deepest bend)
    samples: int = 0
    sum_angle: float = 0.0

    def update(self, knee_angle: float) -> None:
        if math.isnan(knee_angle):
            return
        self.peak_extension = max(self.peak_extension, knee_angle)
        self.min_flexion = min(self.min_flexion, knee_angle)
        self.sum_angle += knee_angle
        self.samples += 1

    @property
    def mean_angle(self) -> float:
        return self.sum_angle / self.samples if self.samples else float("nan")


@dataclass
class GaitAnalyzer:
    """Accumulates running kinematics frame-by-frame."""

    fps: float = 30.0
    osc_window: int = 90  # rolling baseline window (~3 s at 30 fps)

    left: SideStats = field(default_factory=lambda: SideStats("Left"))
    right: SideStats = field(default_factory=lambda: SideStats("Right"))

    _hip_y_history: deque = field(default_factory=lambda: deque(maxlen=90))
    _torso_offsets: list = field(default_factory=list)
    _osc_amplitudes: list = field(default_factory=list)
    frames: int = 0

    def __post_init__(self) -> None:
        self._hip_y_history = deque(maxlen=self.osc_window)

    # -- per-frame metric helpers -------------------------------------------

    def knee_angle(self, kps, conf, side: str):
        hip, knee, ankle = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
        idx = [KP[hip], KP[knee], KP[ankle]]
        if any(conf[i] < KP_CONF for i in idx):
            return None
        return angle_law_of_cosines(kps[idx[0]], kps[idx[1]], kps[idx[2]])

    def hip_centroid(self, kps, conf):
        li, ri = KP["l_hip"], KP["r_hip"]
        if conf[li] < KP_CONF or conf[ri] < KP_CONF:
            return None
        return midpoint(kps[li], kps[ri])

    def torso_offset(self, kps, conf):
        """Angular offset (deg) between shoulder midline and hip midline."""
        ls, rs, lh, rh = (
            KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"],
        )
        if any(conf[i] < KP_CONF for i in (ls, rs, lh, rh)):
            return None
        shoulder_angle = line_angle_deg(kps[ls], kps[rs])
        hip_angle = line_angle_deg(kps[lh], kps[rh])
        diff = (shoulder_angle - hip_angle + 180.0) % 360.0 - 180.0
        return abs(diff)

    def vertical_oscillation(self, hip_y: float):
        """Rolling-variance baseline of hip vertical displacement (pixels)."""
        self._hip_y_history.append(hip_y)
        if len(self._hip_y_history) < 5:
            return None
        arr = np.asarray(self._hip_y_history, dtype=np.float64)
        # Peak-to-peak amplitude around the rolling baseline.
        amplitude = float(arr.max() - arr.min())
        std = float(arr.std())
        return amplitude, std

    # -- main update --------------------------------------------------------

    def update(self, kps, conf):
        """Process one runner's keypoints. Returns per-frame overlay metrics."""
        self.frames += 1
        metrics = {}

        l_knee = self.knee_angle(kps, conf, "l")
        r_knee = self.knee_angle(kps, conf, "r")
        if l_knee is not None:
            self.left.update(l_knee)
            metrics["l_knee"] = l_knee
        if r_knee is not None:
            self.right.update(r_knee)
            metrics["r_knee"] = r_knee

        centroid = self.hip_centroid(kps, conf)
        if centroid is not None:
            osc = self.vertical_oscillation(centroid[1])
            if osc is not None:
                amplitude, std = osc
                self._osc_amplitudes.append(amplitude)
                metrics["osc_amp"] = amplitude
                metrics["osc_std"] = std
            metrics["hip_centroid"] = centroid

        offset = self.torso_offset(kps, conf)
        if offset is not None:
            self._torso_offsets.append(offset)
            metrics["torso_offset"] = offset

        return metrics

    # -- summary ------------------------------------------------------------

    def asymmetry_pct(self) -> float:
        lp, rp = self.left.peak_extension, self.right.peak_extension
        denom = max(lp, rp)
        return abs(lp - rp) / denom * 100.0 if denom else float("nan")

    def report(self) -> str:
        mean_osc = (
            float(np.mean(self._osc_amplitudes)) if self._osc_amplitudes else float("nan")
        )
        mean_torso = (
            float(np.mean(self._torso_offsets)) if self._torso_offsets else float("nan")
        )
        asym = self.asymmetry_pct()

        lines = [
            "",
            "=" * 60,
            "  BIOMECHANICAL EFFICIENCY BREAKDOWN",
            "=" * 60,
            f"  Frames analyzed............. {self.frames}",
            "",
            "  KNEE FLEXION / EXTENSION (deg)",
            f"    Left  : peak ext {self.left.peak_extension:6.1f} | "
            f"min flex {self.left.min_flexion:6.1f} | mean {self.left.mean_angle:6.1f}",
            f"    Right : peak ext {self.right.peak_extension:6.1f} | "
            f"min flex {self.right.min_flexion:6.1f} | mean {self.right.mean_angle:6.1f}",
            "",
            "  VERTICAL OSCILLATION (hip centroid)",
            f"    Mean bounce amplitude...... {mean_osc:6.1f} px",
            "",
            "  PELVIC / TORSO STABILITY",
            f"    Mean shoulder-hip offset... {mean_torso:6.1f} deg",
            "",
            "  LEFT vs RIGHT ASYMMETRY",
            f"    Peak-extension asymmetry... {asym:6.1f} %",
            f"    {self._asymmetry_verdict(asym)}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _asymmetry_verdict(asym: float) -> str:
        if math.isnan(asym):
            return "Insufficient data for asymmetry assessment."
        if asym < 3.0:
            return "Balanced gait - no significant compensation detected."
        if asym < 8.0:
            return "Mild asymmetry - monitor for developing compensation."
        return "Significant asymmetry - possible injury/compensation pattern."


# COCO skeleton edges for drawing.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def draw_overlay(frame, kps, conf, metrics):
    """Draw skeleton plus active joint angles / metrics onto the frame."""
    # Skeleton edges.
    for a, b in SKELETON:
        if conf[a] >= KP_CONF and conf[b] >= KP_CONF:
            pa = tuple(map(int, kps[a]))
            pb = tuple(map(int, kps[b]))
            cv2.line(frame, pa, pb, (0, 255, 0), 2)

    # Keypoints.
    for i, (x, y) in enumerate(kps):
        if conf[i] >= KP_CONF:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

    # Active knee angles at the knee joints.
    for side, idx in (("l", KP["l_knee"]), ("r", KP["r_knee"])):
        key = f"{side}_knee"
        if key in metrics and conf[idx] >= KP_CONF:
            x, y = kps[idx]
            cv2.putText(
                frame, f"{metrics[key]:.0f}", (int(x) + 6, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

    # Hip centroid marker.
    if "hip_centroid" in metrics:
        cx, cy = map(int, metrics["hip_centroid"])
        cv2.drawMarker(frame, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)

    # HUD panel.
    hud = [
        f"L knee: {metrics.get('l_knee', float('nan')):.0f} deg",
        f"R knee: {metrics.get('r_knee', float('nan')):.0f} deg",
        f"Vert osc: {metrics.get('osc_amp', float('nan')):.0f} px",
        f"Torso offset: {metrics.get('torso_offset', float('nan')):.0f} deg",
    ]
    y0 = 28
    for i, text in enumerate(hud):
        cv2.putText(
            frame, text, (12, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame


def load_model(name: str) -> "YOLO":
    try:
        return YOLO(name)
    except Exception as exc:  # noqa: BLE001 - fall back to a stable checkpoint
        if name != FALLBACK_MODEL:
            print(f"[warn] could not load {name} ({exc}); falling back to {FALLBACK_MODEL}")
            return YOLO(FALLBACK_MODEL)
        raise


def analyze(source: str, output: str | None, model_name: str, show: bool) -> None:
    model = load_model(model_name)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    analyzer = GaitAnalyzer(fps=fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, verbose=False)
        result = results[0]

        if result.keypoints is not None and len(result.keypoints) > 0:
            # Track the most prominent runner (largest detection / first person).
            kps_all = result.keypoints.xy.cpu().numpy()
            conf_all = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(kps_all.shape[:2])
            )
            kps = kps_all[0]
            conf = conf_all[0]
            metrics = analyzer.update(kps, conf)
            frame = draw_overlay(frame, kps, conf, metrics)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("Gait Analyzer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[info] annotated video written to {output}")
    if show:
        cv2.destroyAllWindows()

    print(analyzer.report())


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Distance-running gait & biomechanics analyzer (YOLO pose)."
    )
    p.add_argument("--source", required=True, help="Path/URL to the running video.")
    p.add_argument("--output", default=None, help="Optional annotated output video path.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="YOLO pose checkpoint.")
    p.add_argument("--show", action="store_true", help="Display annotated frames live.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    analyze(args.source, args.output, args.model, args.show)


if __name__ == "__main__":
    main()
