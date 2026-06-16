"""Turn per-frame pose keypoints into gait metrics.

This is the analysis core that sits between the pose tracker and the
economy model. It is deliberately pure-Python (no numpy/opencv) so it
runs and tests anywhere; the heavy video + YOLO work lives in
``video.py`` and feeds keypoint samples into here.

Input model
-----------
A *segment* is one continuous stretch of footage in which a single
tracked athlete is visible (one camera shot, no cut). It is a list of
``Sample``s, each a timestamp plus a dict of COCO-style keypoint name ->
(x, y) pixel coordinates. Broadcast coverage cuts between cameras and
loses the athlete, so a race is a *list of segments*; metrics are
computed per segment and then averaged with duration weighting. That
average IS the "assume the average when she's off-screen" rule: time we
never saw simply inherits the behaviour we did see.

Everything here is an estimate. Pixel measurements depend on camera
angle and need calibration (see ``pixel_scale_cm``); treat the outputs
as informed readings, not lab data.
"""

from dataclasses import dataclass
from statistics import median

# COCO-17 keypoint names we rely on.
NOSE = "nose"
L_HIP, R_HIP = "left_hip", "right_hip"
L_ANKLE, R_ANKLE = "left_ankle", "right_ankle"


@dataclass
class Sample:
    t: float                      # seconds
    kp: dict                      # name -> (x, y) in pixels


def _present(sample, *names):
    return all(sample.kp.get(n) is not None for n in names)


def _midpoint(sample, a, b):
    (ax, ay), (bx, by) = sample.kp[a], sample.kp[b]
    return (ax + bx) / 2.0, (ay + by) / 2.0


def _com(sample):
    """Center-of-mass proxy: midpoint of the hips."""
    return _midpoint(sample, L_HIP, R_HIP)


def _runs(flags):
    """Yield (start, end) index pairs of contiguous True runs."""
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            yield start, i
            start = None
    if start is not None:
        yield start, len(flags)


def pixel_scale_cm(samples, athlete_height_cm):
    """cm-per-pixel from the athlete's nose-to-ankle pixel length.

    Uses the median over frames where nose and an ankle are both visible
    and the body is reasonably upright. Nose-to-ankle is ~0.88 of full
    standing height, so we scale by that. Rough, but stable.
    """
    lengths = []
    for s in samples:
        if not _present(s, NOSE):
            continue
        ankle = None
        if _present(s, L_ANKLE):
            ankle = s.kp[L_ANKLE]
        elif _present(s, R_ANKLE):
            ankle = s.kp[R_ANKLE]
        if ankle is None:
            continue
        dy = abs(s.kp[NOSE][1] - ankle[1])
        if dy > 0:
            lengths.append(dy)
    if not lengths:
        return None
    nose_to_ankle_px = median(lengths)
    nose_to_ankle_cm = 0.88 * athlete_height_cm
    return nose_to_ankle_cm / nose_to_ankle_px


def cadence_and_contact(samples, fps, contact_band=0.6):
    """Estimate cadence (steps/min) and mean ground-contact time (ms).

    For each foot we track the ankle height *relative to the hips* (which
    cancels most camera panning). The foot is judged 'planted' while that
    relative height sits in the lower ``contact_band`` of its range; each
    planted run is one ground contact, and its start is one foot strike.
    """
    if len(samples) < 4:
        return None, None
    duration = samples[-1].t - samples[0].t
    if duration <= 0:
        return None, None

    strikes = 0
    contact_frames = []
    for ankle in (L_ANKLE, R_ANKLE):
        rel = [
            (s.t, s.kp[ankle][1] - _com(s)[1])
            for s in samples
            if _present(s, ankle, L_HIP, R_HIP)
        ]
        if len(rel) < 4:
            continue
        ys = [y for _, y in rel]
        lo, hi = min(ys), max(ys)
        if hi - lo <= 0:
            continue
        threshold = lo + contact_band * (hi - lo)
        planted = [y >= threshold for y in ys]   # larger y = lower = planted
        for start, end in _runs(planted):
            strikes += 1
            contact_frames.append(end - start)

    if not strikes:
        return None, None
    cadence = strikes / (duration / 60.0)
    mean_contact_ms = 1000.0 * (sum(contact_frames) / len(contact_frames)) / fps
    return cadence, mean_contact_ms


def _amplitude(values):
    """Mean peak-to-trough amplitude of a 1-D signal."""
    extrema = []
    for i in range(1, len(values) - 1):
        a, b, c = values[i - 1], values[i], values[i + 1]
        if (b >= a and b >= c) or (b <= a and b <= c):
            extrema.append(b)
    if len(extrema) < 2:
        return max(values) - min(values) if values else 0.0
    swings = [abs(extrema[i + 1] - extrema[i]) for i in range(len(extrema) - 1)]
    return sum(swings) / len(swings)


def vertical_oscillation_cm(samples, scale_cm_per_px):
    com_y = [_com(s)[1] for s in samples if _present(s, L_HIP, R_HIP)]
    if len(com_y) < 3 or scale_cm_per_px is None:
        return None
    return _amplitude(com_y) * scale_cm_per_px


def head_sway_cm(samples, scale_cm_per_px):
    """Side-to-side head travel, with forward translation removed."""
    rel_x = [
        s.kp[NOSE][0] - _com(s)[0]
        for s in samples
        if _present(s, NOSE, L_HIP, R_HIP)
    ]
    if len(rel_x) < 3 or scale_cm_per_px is None:
        return None
    return _amplitude(rel_x) * scale_cm_per_px


def metrics_for_segment(samples, fps, athlete_height_cm):
    """All four economy-relevant metrics for one continuous segment."""
    scale = pixel_scale_cm(samples, athlete_height_cm)
    cadence, contact = cadence_and_contact(samples, fps)
    return {
        "ground_contact_ms": contact,
        "cadence_spm": cadence,
        "vertical_osc_cm": vertical_oscillation_cm(samples, scale),
        "head_sway_cm": head_sway_cm(samples, scale),
    }


COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def segments_from_json(obj):
    """Build (segments, fps, height_cm) from a poses JSON document.

    Schema::

        {"fps": 50, "athlete_height_cm": 165,
         "segments": [[{"t": 0.0, "kp": {"nose": [x, y], ...}}, ...], ...]}

    A keypoint may be null/absent when not detected in that frame.
    """
    fps = obj["fps"]
    height_cm = obj["athlete_height_cm"]
    segments = []
    for seg in obj["segments"]:
        samples = []
        for rec in seg:
            kp = {k: (tuple(v) if v else None) for k, v in rec.get("kp", {}).items()}
            samples.append(Sample(t=rec["t"], kp=kp))
        segments.append(samples)
    return segments, fps, height_cm


def aggregate_segments(segment_metrics, weights=None):
    """Duration-weighted average of per-segment metrics, skipping gaps.

    `segment_metrics` is a list of dicts from `metrics_for_segment`;
    `weights` (e.g. each segment's visible duration) defaults to equal.
    A metric missing in a segment is simply left out of its average --
    the remaining segments stand in for the unseen time.
    """
    if weights is None:
        weights = [1.0] * len(segment_metrics)
    out = {}
    for key in ("ground_contact_ms", "cadence_spm", "vertical_osc_cm", "head_sway_cm"):
        num = den = 0.0
        for m, w in zip(segment_metrics, weights):
            v = m.get(key)
            if v is not None:
                num += v * w
                den += w
        out[key] = (num / den) if den else None
    return out
