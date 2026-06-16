"""Run pose tracking over a race video and emit keypoint segments.

This is the only module that touches heavy CV dependencies, and it
imports them lazily so the rest of the package (and the test suite) work
without them. To use it locally:

    pip install ultralytics opencv-python

Then, on your machine where the video lives:

    from athlete_predictor.video import extract_segments, save_poses_json

    # 1. Detect every person-track in every camera shot.
    shots = extract_segments("rome_5000m.mp4", model="yolov8n-pose.pt")

    # 2. Tell it which track is your athlete in each shot. Track IDs reset
    #    at every camera cut, so this is where you point at Senayet --
    #    e.g. by eyeballing the preview, or by lane/bib if you automate it.
    chosen = {shot_index: track_id, ...}

    # 3. Keep only her, write the JSON the analyzer consumes.
    save_poses_json("senayet_poses.json", shots, chosen,
                    fps=shots.fps, athlete_height_cm=165)

Then back here (works anywhere, no CV deps):

    python -m athlete_predictor analyze --poses senayet_poses.json \
        --athlete "Senayet Getachew" --event 5000m --gear super_spikes

Camera cuts are detected by frame-to-frame colour-histogram change, and
each shot is tracked independently -- that is what lets the analyzer
average over only the footage where she is actually visible.
"""

import json

from .pose_analysis import COCO_KEYPOINTS


def _lazy_imports():
    try:
        import cv2  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "video analysis needs extra packages: pip install ultralytics opencv-python"
        ) from exc
    return cv2, YOLO


def _histogram(cv2, frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(hist, hist)
    return hist


def detect_shot_boundaries(video_path, threshold=0.5):  # pragma: no cover - needs cv2
    """Frame indices where the camera cuts, via colour-histogram distance.

    Standalone helper. ``extract_segments`` does this inline in its single
    pass instead, so you rarely need to call this separately.
    """
    cv2, _ = _lazy_imports()
    cap = cv2.VideoCapture(video_path)
    boundaries, prev_hist, idx = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist = _histogram(cv2, frame)
        if prev_hist is not None:
            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist > threshold:
                boundaries.append(idx)
        prev_hist, idx = hist, idx + 1
    cap.release()
    return boundaries


def extract_segments(video_path, model="yolov8s-pose.pt", conf=0.4, imgsz=640,
                     vid_stride=1, cut_threshold=0.5, preview_offset=10,
                     max_frames=None, device=None, half=False,
                     progress=True):  # pragma: no cover
    """Pose-track every person per camera shot, in a single decode pass.

    Speed knobs (all matter on a T4):
      * ``model``      -- 'yolov8n-pose.pt' (fastest) .. 'yolov8x-pose.pt' (best).
      * ``imgsz``      -- inference resolution; 480 or 384 is much faster than 640.
      * ``vid_stride`` -- process every Nth frame (2 ~halves the work). Timestamps
                          and contact-time stay correct because the reported fps is
                          divided by the stride.
      * ``max_frames`` -- cap frames for a quick trial run on a short slice first.

    Returns an object with ``.fps``, ``.shots`` (one dict per shot mapping
    track_id -> list of Samples) and ``.previews`` (one annotated frame per
    shot). Shot cuts are detected inline from the frames YOLO already
    decodes, so the video is read only once.
    """
    import time

    from .pose_analysis import Sample

    cv2, YOLO = _lazy_imports()
    net = YOLO(model)

    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    opened = cap.isOpened()
    cap.release()
    eff_fps = real_fps / max(1, vid_stride)
    if progress:
        print(f"reading {video_path}: {total} frames @ {real_fps:.0f} fps")
    if not opened or total <= 0:
        raise RuntimeError(
            f"{video_path!r} has no decodable frames (opened={opened}, frames={total}). "
            "The download almost certainly failed -- check the download cell's output "
            "and that the file size is more than a few KB."
        )

    shots, previews, current = [], [], {}
    prev_hist, pending_preview, shot_local_idx = None, None, 0
    processed = 0
    start = time.time()

    stream = net.track(
        video_path, stream=True, conf=conf, imgsz=imgsz, vid_stride=vid_stride,
        persist=True, verbose=False, device=device, half=half,
    )
    for idx, result in enumerate(stream):
        if max_frames and idx >= max_frames:
            break
        processed = idx + 1

        hist = _histogram(cv2, result.orig_img)
        is_cut = (
            prev_hist is not None
            and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > cut_threshold
        )
        prev_hist = hist
        if is_cut and current:
            shots.append(current)
            previews.append(pending_preview)
            current, shot_local_idx, pending_preview = {}, 0, None

        if shot_local_idx == preview_offset:
            pending_preview = result.plot()
        shot_local_idx += 1

        if result.keypoints is not None and result.boxes is not None \
                and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            xy = result.keypoints.xy.tolist()
            for tid, points in zip(ids, xy):
                kp = {
                    name: (float(x), float(y)) if (x or y) else None
                    for name, (x, y) in zip(COCO_KEYPOINTS, points)
                }
                current.setdefault(tid, []).append(Sample(t=idx / eff_fps, kp=kp))

        if progress and idx and idx % 50 == 0:
            rate = idx / (time.time() - start)
            pct = f" ({100 * idx * vid_stride / total:.0f}%)" if total else ""
            print(f"  frame {idx}{pct} — {rate:.1f} fps, {len(shots) + 1} shot(s)")

    if current:
        shots.append(current)
        previews.append(pending_preview)

    if not processed:
        raise RuntimeError(
            f"the tracker received 0 frames from {video_path!r} even though its header "
            f"reports {total} frames. The file is likely corrupt or only partially "
            "downloaded -- re-download it (check the download cell's output)."
        )
    if progress:
        print(f"done: {processed} frame(s), {len(shots)} shot(s) in {time.time() - start:.0f}s")
    return _Shots(fps=eff_fps, shots=shots, previews=previews)


class _Shots:
    def __init__(self, fps, shots, previews=None):
        self.fps = fps
        self.shots = shots
        self.previews = previews or []

    def track_ids(self, shot_index):
        """Track IDs visible in a shot, longest-tracked first."""
        tracks = self.shots[shot_index]
        return sorted(tracks, key=lambda tid: -len(tracks[tid]))

    def save_previews(self, out_dir):  # pragma: no cover - needs cv2
        """Write one annotated JPG per shot for picking your athlete."""
        import os

        cv2, _ = _lazy_imports()
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, img in enumerate(self.previews):
            if img is None:
                continue
            path = os.path.join(out_dir, f"shot_{i:03d}.jpg")
            cv2.imwrite(path, img)
            paths.append(path)
        return paths


def save_poses_json(path, shots, chosen, fps, athlete_height_cm):
    """Write the analyzer's poses JSON for one athlete.

    `chosen` maps shot index -> the track_id that is your athlete in that
    shot. Shots where she does not appear are simply omitted.
    """
    segments = []
    for shot_index, track_id in sorted(chosen.items()):
        samples = shots.shots[shot_index][track_id]
        segments.append([
            {"t": s.t, "kp": {k: list(v) if v else None for k, v in s.kp.items()}}
            for s in samples
        ])
    doc = {"fps": fps, "athlete_height_cm": athlete_height_cm, "segments": segments}
    with open(path, "w") as f:
        json.dump(doc, f)
    return doc
