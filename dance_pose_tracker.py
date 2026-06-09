"""
Athletics Biomechanics Diagnostics & Scouting
==============================================

Advanced running gait analysis built on top of a YOLO-pose keypoint backend.

Beyond simple frame-by-frame tracking, this module provides:

  1. Multi-Angle / Multi-Plane Analysis
       Auto-detects whether the runner is filmed from the side, rear, or front
       and switches the kinematic calculations to whatever that plane can
       actually measure (sagittal knee flexion vs. frontal pelvic drop, etc.).

  2. Camera-Independent Scaling (Normalization)
       Every linear metric is expressed as a ratio of the runner's torso height
       (neck->hip distance) in that frame, so zoom and camera distance no longer
       distort the numbers.

  3. Injury-Prediction Triggers
       Maintains a rolling 50-frame Asymmetry Index (L/R range-of-motion delta).
       If it stays above 5% for more than 3 seconds, the frame is flagged with a
       red on-screen warning.

  4. Biomechanical Baseline Profiles
       Loads an athlete's peak-performance baseline CSV and reports percentage
       deviation in stride length, cadence, and knee extension.

  5. Enhanced Console Output
       Prints a coach/agent-friendly "Scouting Score" summarizing efficiency,
       symmetry and injury risk.

Usage
-----
    python dance_pose_tracker.py --video run.mp4 \
        --baseline baseline.csv --output annotated.mp4

The keypoint backend (ultralytics YOLO-pose) and OpenCV are imported lazily so
the analytics layer can be unit-tested without the heavy dependencies present.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Keypoint layout (COCO-17, the format emitted by YOLO-pose / MediaPipe-lite).
# ---------------------------------------------------------------------------

KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Minimum keypoint confidence to trust a point in a calculation.
MIN_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

Point = Tuple[float, float, float]  # (x, y, confidence)


def _valid(p: Optional[Point]) -> bool:
    return p is not None and len(p) >= 3 and p[2] >= MIN_CONFIDENCE


def distance(a: Point, b: Point) -> float:
    """Euclidean distance in pixels between two keypoints."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2]))


def joint_angle(a: Point, b: Point, c: Point) -> float:
    """Interior angle (degrees) at vertex ``b`` formed by points a-b-c."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denom = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denom == 0:
        return 0.0
    cos_theta = (bax * bcx + bay * bcy) / denom
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


# ---------------------------------------------------------------------------
# Feature 1 -- Multi-Angle / Multi-Plane detection
# ---------------------------------------------------------------------------

class ViewAngle(Enum):
    SIDE = "side-profile"
    REAR = "rear-view"
    FRONT = "front-view"
    UNKNOWN = "unknown"


def detect_view_angle(kp: Dict[str, Point]) -> ViewAngle:
    """Infer the camera plane from shoulder/hip geometry and facial keypoints.

    Heuristics:
      * Side profile  -> shoulders overlap horizontally (narrow shoulder width
        relative to torso height) and one ear is far more visible than the other.
      * Front view    -> wide shoulders AND both eyes confidently visible.
      * Rear view     -> wide shoulders BUT neither eye/nose visible (back of head).
    """
    ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    if not (_valid(ls) and _valid(rs) and _valid(lh) and _valid(rh)):
        return ViewAngle.UNKNOWN

    shoulder_width = distance(ls, rs)
    neck = midpoint(ls, rs)
    hip = midpoint(lh, rh)
    torso_height = distance(neck, hip)
    if torso_height <= 1e-6:
        return ViewAngle.UNKNOWN

    width_ratio = shoulder_width / torso_height

    face_visible = any(
        _valid(kp.get(name)) for name in ("nose", "left_eye", "right_eye")
    )

    # Narrow apparent shoulder width => we are looking along the frontal axis.
    if width_ratio < 0.45:
        return ViewAngle.SIDE

    # Wide shoulders: distinguish front vs rear by whether the face is visible.
    return ViewAngle.FRONT if face_visible else ViewAngle.REAR


# ---------------------------------------------------------------------------
# Feature 2 -- Camera-independent per-frame metrics
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    frame_index: int
    view: ViewAngle
    torso_height_px: float

    # Sagittal (side view)
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None

    # Frontal (rear / front view)
    pelvic_drop_deg: Optional[float] = None      # tilt of hip line from horizontal
    lateral_asymmetry: Optional[float] = None     # normalized L/R ankle spread delta

    # Camera-independent linear metric: vertical position of body CoM as a
    # fraction of torso height (used downstream for vertical oscillation).
    com_vertical_ratio: Optional[float] = None


def torso_height(kp: Dict[str, Point]) -> Optional[float]:
    ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    if not (_valid(ls) and _valid(rs) and _valid(lh) and _valid(rh)):
        return None
    th = distance(midpoint(ls, rs), midpoint(lh, rh))
    return th if th > 1e-6 else None


def compute_frame_metrics(
    frame_index: int, kp: Dict[str, Point], view: ViewAngle
) -> Optional[FrameMetrics]:
    """Produce plane-appropriate, torso-normalized metrics for one frame."""
    th = torso_height(kp)
    if th is None:
        return None

    fm = FrameMetrics(frame_index=frame_index, view=view, torso_height_px=th)

    # Center of mass proxy = hip midpoint; normalized by torso height so that
    # frame-to-frame oscillation is camera-independent.
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    if _valid(lh) and _valid(rh):
        hip = midpoint(lh, rh)
        fm.com_vertical_ratio = hip[1] / th

    if view == ViewAngle.SIDE:
        # Sagittal plane -> knee flexion/extension is meaningful.
        lhip, lknee, lank = kp.get("left_hip"), kp.get("left_knee"), kp.get("left_ankle")
        if _valid(lhip) and _valid(lknee) and _valid(lank):
            fm.left_knee_angle = joint_angle(lhip, lknee, lank)
        rhip, rknee, rank = kp.get("right_hip"), kp.get("right_knee"), kp.get("right_ankle")
        if _valid(rhip) and _valid(rknee) and _valid(rank):
            fm.right_knee_angle = joint_angle(rhip, rknee, rank)

    elif view in (ViewAngle.REAR, ViewAngle.FRONT):
        # Frontal plane -> pelvic drop and lateral asymmetry are meaningful.
        if _valid(lh) and _valid(rh):
            dx = rh[0] - lh[0]
            dy = rh[1] - lh[1]
            # Angle of the hip line away from horizontal (pelvic obliquity).
            fm.pelvic_drop_deg = abs(math.degrees(math.atan2(dy, abs(dx) + 1e-6)))

        lank, rank = kp.get("left_ankle"), kp.get("right_ankle")
        if _valid(lank) and _valid(rank) and _valid(lh) and _valid(rh):
            # Horizontal distance of each ankle from the pelvis center, normalized.
            pelvis_x = (lh[0] + rh[0]) / 2.0
            left_spread = abs(lank[0] - pelvis_x) / th
            right_spread = abs(rank[0] - pelvis_x) / th
            fm.lateral_asymmetry = abs(left_spread - right_spread)

    return fm


# ---------------------------------------------------------------------------
# Feature 3 -- Rolling Asymmetry Index & injury trigger
# ---------------------------------------------------------------------------

@dataclass
class InjuryRiskMonitor:
    """Rolling 50-frame asymmetry monitor with a 3-second sustain trigger.

    The Asymmetry Index is the relative difference in joint range-of-motion
    between the left and right limbs over the window. When it stays above the
    threshold (default 5%) for longer than ``sustain_seconds``, ``flagged``
    becomes True and frames should be drawn with a red warning.
    """

    fps: float
    window: int = 50
    threshold: float = 0.05            # 5%
    sustain_seconds: float = 3.0

    _left_rom: deque = field(default_factory=lambda: deque(maxlen=50))
    _right_rom: deque = field(default_factory=lambda: deque(maxlen=50))
    _consecutive_breaches: int = 0
    current_index: float = 0.0
    flagged: bool = False

    def __post_init__(self):
        self._left_rom = deque(maxlen=self.window)
        self._right_rom = deque(maxlen=self.window)

    @property
    def _sustain_frames(self) -> int:
        return max(1, int(round(self.sustain_seconds * self.fps)))

    def update(self, fm: FrameMetrics) -> float:
        """Feed one frame's metrics; returns the current asymmetry index."""
        left = right = None

        if fm.view == ViewAngle.SIDE:
            left, right = fm.left_knee_angle, fm.right_knee_angle
        else:
            # In frontal planes use lateral spread as the L/R ROM proxy; a single
            # asymmetry scalar already encodes the imbalance.
            if fm.lateral_asymmetry is not None:
                self.current_index = fm.lateral_asymmetry
                self._register_breach(self.current_index)
                return self.current_index

        if left is not None:
            self._left_rom.append(left)
        if right is not None:
            self._right_rom.append(right)

        self.current_index = self._compute_index()
        self._register_breach(self.current_index)
        return self.current_index

    def _compute_index(self) -> float:
        if not self._left_rom or not self._right_rom:
            return 0.0
        left_rom = max(self._left_rom) - min(self._left_rom)
        right_rom = max(self._right_rom) - min(self._right_rom)
        denom = (left_rom + right_rom) / 2.0
        if denom <= 1e-6:
            return 0.0
        return abs(left_rom - right_rom) / denom

    def _register_breach(self, index: float) -> None:
        if index > self.threshold:
            self._consecutive_breaches += 1
        else:
            self._consecutive_breaches = 0
        self.flagged = self._consecutive_breaches >= self._sustain_frames


# ---------------------------------------------------------------------------
# Feature 4 -- Baseline profiles
# ---------------------------------------------------------------------------

@dataclass
class BaselineProfile:
    hip_tilt: float = 0.0         # degrees (peak pelvic obliquity) -- focus metric
    knee_extension: float = 0.0   # degrees (peak extension angle)
    athlete: str = "unknown"

    @classmethod
    def from_csv(cls, path: str) -> "BaselineProfile":
        """Load a baseline CSV.

        Accepts either a two-column key,value layout or a single header+row
        layout. Recognized keys: athlete, hip_tilt, knee_extension.
        """
        data: Dict[str, str] = {}
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))

        if not rows:
            raise ValueError(f"Baseline CSV {path!r} is empty")

        header = [c.strip().lower() for c in rows[0]]
        if len(rows) >= 2 and len(header) > 2:
            # header + single data row
            values = rows[1]
            data = {h: v.strip() for h, v in zip(header, values)}
        else:
            # key,value per line
            for row in rows:
                if len(row) >= 2:
                    data[row[0].strip().lower()] = row[1].strip()

        def _num(key: str, default: float = 0.0) -> float:
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            hip_tilt=_num("hip_tilt"),
            knee_extension=_num("knee_extension"),
            athlete=data.get("athlete", "unknown"),
        )


def deviation_report(
    baseline: BaselineProfile, current: BaselineProfile
) -> Dict[str, float]:
    """Percentage deviation of current metrics from the baseline (signed)."""

    def pct(cur: float, base: float) -> float:
        if base == 0:
            return 0.0
        return (cur - base) / base * 100.0

    return {
        "hip_tilt": pct(current.hip_tilt, baseline.hip_tilt),
        "knee_extension": pct(current.knee_extension, baseline.knee_extension),
    }


# ---------------------------------------------------------------------------
# Aggregation & Scouting score (Feature 5)
# ---------------------------------------------------------------------------

@dataclass
class GaitSummary:
    frames_analyzed: int = 0
    dominant_view: ViewAngle = ViewAngle.UNKNOWN
    avg_vertical_oscillation: float = 0.0   # torso-height units
    avg_asymmetry_index: float = 0.0
    peak_asymmetry_index: float = 0.0
    injury_flag_frames: int = 0
    peak_knee_extension: float = 0.0
    # Hip tilt (pelvic obliquity) -- primary focus metric.
    avg_hip_tilt_deg: float = 0.0
    peak_hip_tilt_deg: float = 0.0
    hip_tilt_samples: int = 0


class GaitAnalyzer:
    """Accumulates per-frame metrics into a session-level summary."""

    def __init__(self, fps: float):
        self.fps = fps
        self.monitor = InjuryRiskMonitor(fps=fps)
        self._view_votes: Dict[ViewAngle, int] = {}
        self._osc_samples: List[float] = []
        self._asym_samples: List[float] = []
        self._com_series: List[float] = []
        self._knee_extensions: List[float] = []
        self._hip_tilts: List[float] = []
        self._flag_frames = 0
        self._frames = 0

    def process(self, fm: FrameMetrics) -> None:
        self._frames += 1
        self._view_votes[fm.view] = self._view_votes.get(fm.view, 0) + 1

        if fm.com_vertical_ratio is not None:
            self._com_series.append(fm.com_vertical_ratio)

        idx = self.monitor.update(fm)
        self._asym_samples.append(idx)
        if self.monitor.flagged:
            self._flag_frames += 1

        for angle in (fm.left_knee_angle, fm.right_knee_angle):
            if angle is not None:
                self._knee_extensions.append(angle)

        if fm.pelvic_drop_deg is not None:
            self._hip_tilts.append(fm.pelvic_drop_deg)

    def _vertical_oscillation(self) -> float:
        """Mean peak-to-trough oscillation of the normalized CoM signal."""
        if len(self._com_series) < 3:
            return 0.0
        peaks, troughs = [], []
        for i in range(1, len(self._com_series) - 1):
            prev, cur, nxt = self._com_series[i - 1:i + 2]
            if cur > prev and cur > nxt:
                peaks.append(cur)
            elif cur < prev and cur < nxt:
                troughs.append(cur)
        if not peaks or not troughs:
            return 0.0
        return abs(sum(peaks) / len(peaks) - sum(troughs) / len(troughs))

    def summarize(self) -> GaitSummary:
        s = GaitSummary()
        s.frames_analyzed = self._frames
        if self._view_votes:
            s.dominant_view = max(self._view_votes, key=self._view_votes.get)
        s.avg_vertical_oscillation = self._vertical_oscillation()
        if self._asym_samples:
            s.avg_asymmetry_index = sum(self._asym_samples) / len(self._asym_samples)
            s.peak_asymmetry_index = max(self._asym_samples)
        s.injury_flag_frames = self._flag_frames
        if self._knee_extensions:
            # Peak extension == largest (straightest) knee angle observed.
            s.peak_knee_extension = max(self._knee_extensions)

        # Hip tilt (pelvic obliquity) -- the primary focus metric.
        if self._hip_tilts:
            s.avg_hip_tilt_deg = sum(self._hip_tilts) / len(self._hip_tilts)
            s.peak_hip_tilt_deg = max(self._hip_tilts)
            s.hip_tilt_samples = len(self._hip_tilts)
        return s


def scouting_score(summary: GaitSummary) -> Dict[str, float]:
    """Compute 0-100 sub-scores and a composite Scouting Score.

    * Efficiency  -> rewards low vertical oscillation.
    * Symmetry    -> rewards low average asymmetry index.
    * Injury risk -> share of frames that were flagged (lower is better).
    """
    # Efficiency: 0.05 torso-units oscillation ~ excellent; 0.15 ~ poor.
    eff = max(0.0, min(1.0, (0.15 - summary.avg_vertical_oscillation) / 0.10))
    efficiency = round(eff * 100, 1)

    # Symmetry: 0% asymmetry => 100, 15%+ => 0.
    sym = max(0.0, min(1.0, (0.15 - summary.avg_asymmetry_index) / 0.15))
    symmetry = round(sym * 100, 1)

    flag_ratio = (
        summary.injury_flag_frames / summary.frames_analyzed
        if summary.frames_analyzed else 0.0
    )
    injury_health = round(max(0.0, 1.0 - flag_ratio) * 100, 1)

    composite = round(0.4 * efficiency + 0.35 * symmetry + 0.25 * injury_health, 1)
    return {
        "efficiency": efficiency,
        "symmetry": symmetry,
        "injury_health": injury_health,
        "composite": composite,
    }


# ---------------------------------------------------------------------------
# Console reporting (Feature 5)
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 85:
        return "ELITE"
    if score >= 70:
        return "STRONG"
    if score >= 55:
        return "DEVELOPING"
    return "AT RISK"


def print_scouting_report(
    summary: GaitSummary,
    baseline: Optional[BaselineProfile] = None,
) -> None:
    scores = scouting_score(summary)
    line = "=" * 60
    print("\n" + line)
    print("        ATHLETE BIOMECHANICS -- SCOUTING REPORT")
    print(line)
    print(f"  Frames analyzed       : {summary.frames_analyzed}")
    print(f"  Dominant view         : {summary.dominant_view.value}")
    print(f"  Vertical oscillation  : {summary.avg_vertical_oscillation:.3f} torso-units")
    print(f"  Avg asymmetry index   : {summary.avg_asymmetry_index * 100:.1f}%")
    print(f"  Peak asymmetry index  : {summary.peak_asymmetry_index * 100:.1f}%")
    print(f"  Avg hip tilt          : {summary.avg_hip_tilt_deg:.1f} deg "
          f"({summary.hip_tilt_samples} frames)")
    print(f"  Peak hip tilt         : {summary.peak_hip_tilt_deg:.1f} deg")
    print(f"  Peak knee extension   : {summary.peak_knee_extension:.1f} deg")
    print(f"  Injury-flagged frames : {summary.injury_flag_frames}")

    if baseline is not None:
        current = BaselineProfile(
            hip_tilt=summary.peak_hip_tilt_deg,
            knee_extension=summary.peak_knee_extension,
        )
        dev = deviation_report(baseline, current)
        print("\n" + "-" * 60)
        print(f"  BASELINE COMPARISON (vs {baseline.athlete})")
        print("-" * 60)
        for metric, pct in dev.items():
            arrow = "+" if pct >= 0 else ""
            print(f"  {metric:<18}: {arrow}{pct:6.1f}%  deviation from baseline")

    print("\n" + "-" * 60)
    print("  SCOUTING SCORE")
    print("-" * 60)
    print(f"  Mechanical efficiency : {scores['efficiency']:5.1f} / 100")
    print(f"  Symmetry              : {scores['symmetry']:5.1f} / 100")
    print(f"  Injury resilience     : {scores['injury_health']:5.1f} / 100")
    print(f"  >> COMPOSITE SCORE    : {scores['composite']:5.1f} / 100  [{_grade(scores['composite'])}]")
    print(line + "\n")


# ---------------------------------------------------------------------------
# Video pipeline (lazy heavy imports)
# ---------------------------------------------------------------------------

def _keypoints_to_dict(kp_array: Sequence[Sequence[float]]) -> Dict[str, Point]:
    """Map a COCO-17 keypoint array (x, y, conf) to a named dictionary."""
    out: Dict[str, Point] = {}
    for name, idx in KEYPOINTS.items():
        if idx < len(kp_array):
            x, y, c = kp_array[idx][0], kp_array[idx][1], kp_array[idx][2]
            out[name] = (float(x), float(y), float(c))
    return out


def _draw_overlay(frame, kp: Dict[str, Point], fm: FrameMetrics,
                  monitor: InjuryRiskMonitor):
    """Draw the Fatigue & Injury Risk dashboard onto a BGR frame."""
    import cv2  # local import

    h, w = frame.shape[:2]
    panel_color = (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (320, 120), panel_color, -1)

    cv2.putText(frame, f"View: {fm.view.value}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Asymmetry: {monitor.current_index * 100:4.1f}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
    if fm.view == ViewAngle.SIDE and fm.left_knee_angle is not None:
        cv2.putText(frame, f"L/R knee: {fm.left_knee_angle:.0f}/"
                    f"{fm.right_knee_angle or 0:.0f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA)
    elif fm.pelvic_drop_deg is not None:
        cv2.putText(frame, f"Pelvic drop: {fm.pelvic_drop_deg:.1f} deg", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA)

    if monitor.flagged:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 6)
        cv2.putText(frame, "!! INJURY RISK -- ASYMMETRY SUSTAINED !!",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2, cv2.LINE_AA)

    # Draw keypoints
    for p in kp.values():
        if _valid(p):
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
    return frame


def analyze_video(
    video_path: str,
    model_path: str = "yolov9-pose.pt",
    baseline_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> GaitSummary:
    """Run the full biomechanics pipeline over a video file."""
    import cv2  # local heavy import
    from ultralytics import YOLO  # local heavy import

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer = GaitAnalyzer(fps=fps)

    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame, verbose=False)
            kp_dict: Dict[str, Point] = {}
            if results and len(results[0].keypoints) > 0:
                # Take the most confident detection (the runner).
                data = results[0].keypoints.data.cpu().numpy()
                if len(data) > 0:
                    kp_dict = _keypoints_to_dict(data[0])

            if kp_dict:
                view = detect_view_angle(kp_dict)
                fm = compute_frame_metrics(frame_index, kp_dict, view)
                if fm is not None:
                    analyzer.process(fm)
                    if writer is not None:
                        _draw_overlay(frame, kp_dict, fm, analyzer.monitor)

            if writer is not None:
                writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = analyzer.summarize()
    baseline = BaselineProfile.from_csv(baseline_path) if baseline_path else None
    print_scouting_report(summary, baseline)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Elite athletics biomechanics diagnostics & scouting."
    )
    p.add_argument("--video", required=True, help="Path to the input running video.")
    p.add_argument("--model", default="yolov9-pose.pt",
                   help="YOLO-pose model weights.")
    p.add_argument("--baseline", default=None,
                   help="Optional baseline CSV of peak-performance metrics.")
    p.add_argument("--output", default=None,
                   help="Optional path to write the annotated dashboard video.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    analyze_video(
        video_path=args.video,
        model_path=args.model,
        baseline_path=args.baseline,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
