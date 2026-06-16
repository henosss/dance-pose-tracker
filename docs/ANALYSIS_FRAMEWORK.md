# Biomechanical analysis framework for broadcast 5000m footage

A design note for turning TV race footage into per-athlete gait and
running-economy features. It covers the six problem areas you raised, marks
what is **implemented** in this repo versus what is **architectural guidance**,
and is deliberately honest about where 2D broadcast video can and cannot give
trustworthy numbers.

> The hard truth up front: broadcast footage is uncalibrated, monocular, and
> framed for storytelling, not science. Treat every output as a *relative*
> signal for comparing athletes and tracking change over time, not as
> force-plate truth. The whole framework below is built to maximise relative
> consistency and to refuse to emit a number it can't support.

---

## 1. Shot segmentation

**Why pixel-based scene detection fails here:** a distance race is long
continuous coverage with constant pans and zooms, so a histogram/SAD cut
detector either fires constantly (on pans) or not at all (on slow zooms). Cuts
are the wrong unit anyway — what you want are *clips where your athlete is
trackable*.

**Recommended: segment on athlete presence, not pixels.** Track every person
with a persistent tracker, follow one athlete's ID, and split her timeline
wherever she disappears for longer than a short gap. Each resulting clip is a
"work clip" with a continuous kinematic signal.

- *Implemented:* `pose_analysis.split_on_gaps(samples, max_gap_s)` splits a
  track on presence gaps. `extract_segments` still does a cheap inline
  histogram split as a first cut, but presence-splitting is the robust unit and
  should be the primary segmentation for analysis.
- *Guidance for the next step:* add `PySceneDetect`'s `ContentDetector` only as
  a *hint* (hard cuts between camera feeds), and combine: `new clip` =
  `hard cut` **or** `athlete-presence gap`. Don't let pans trigger splits —
  tune `ContentDetector(threshold=...)` high, or use `AdaptiveDetector` which is
  pan-tolerant.
- *Event markers:* for race-stage labelling (opening / mid / final laps /
  kick), the cleanest signals are the on-screen clock/lap graphics (OCR a fixed
  ROI with `pytesseract`) or bell-lap audio. Lacking those, fall back to a
  fraction-of-race estimate from elapsed time.

## 2. Kinematic metrics

All metrics run on cleaned, side-on segments. **View gating matters:** hip ROM
and GCT are only valid on roughly side-on shots; compute a torso/hip-line
orientation and skip frontal shots (see §5).

### Ground Contact Time (GCT)
2D, monocular GCT is an *estimate of the stance fraction*, not a force-plate
contact time. Two compatible signals:

1. **Vertical position band (implemented).** Track the ankle height relative to
   the hips (cancels most panning). The foot is "planted" while that relative
   height sits in the lower band of its per-stride range; each planted run is
   one contact, its start a foot strike. `cadence_and_contact` does this.
2. **Velocity profile (recommended upgrade).** Compute the ankle's vertical
   velocity `v_y = Δy/Δt`. Stance ≈ the interval bracketed by the foot-strike
   (sharp deceleration / `v_y → 0` at the low point) and toe-off (`v_y` turns
   strongly negative = lifting). Detect strike/toe-off as zero-crossings of a
   smoothed `v_y`. This is more robust to amplitude drift than a fixed band.

`GCT_ms = 1000 * (stance_frames / fps)`; report the per-foot mean and the
flight-to-contact ratio `flight_to_contact_ratio()` — high ratio = the
"floating" stride, low = the heavy, long-contact stride.

### Hip flexibility / sagittal ROM
- *Implemented:* `hip_rom_deg(samples)` = peak-to-peak of the thigh angle
  (hip→knee vector) relative to vertical, `θ = atan2(kx−hx, ky−hy)`, taken over
  the gait cycle, per leg, larger leg returned.
- *Upgrade:* a true hip joint angle uses the trunk segment too: `hip_angle =
  angle(shoulder→hip, hip→knee)`. Report peak flexion (swing) and peak extension
  (toe-off) separately — a stiff, "blocked" hip shows reduced *extension*
  specifically, which is the mechanism behind a long ground contact.
- *Caveat:* sagittal angles need a side-on view; near head-on they collapse to
  noise. Gate on view and on knee/hip keypoint confidence.

### Energy conservation / running economy
Wasted, non-propulsive motion = wasted economy. Quantify it from:
- **Vertical oscillation** of the center-of-mass (hip midpoint),
  `vertical_oscillation_cm` — mean peak-to-trough, calibrated to cm.
- **Center-of-mass lateral stability** — head/torso sway with forward
  translation removed (`head_sway_cm`), and you can add hip lateral sway the
  same way.
- **Cadence** — low cadence at speed implies overstriding/braking.

These map to an economy %-cost vs an elite reference in `biomechanics.py`
(`economy_penalty`), and economy converts to time at ~⅔ transfer in
`economy.py`. The constants are literature-based estimates, exposed for tuning.

## 3. Pose consistency & broadcast robustness (post-processing)

Raw per-frame pose on TV footage is jittery and gappy. The cleanup pass matters
as much as the model:

- *Implemented:* `preprocess()` does per-keypoint **linear interpolation of
  short occlusion gaps** + **median smoothing**; `extract_segments` does
  **confidence gating** (`kp_conf`) so low-confidence joints become missing
  rather than wrong.
- *Recommended additions:*
  - **One-Euro filter** instead of median smoothing for low-lag jitter removal
    on fast limbs.
  - **Camera-motion compensation:** estimate frame-to-frame homography
    (`cv2.estimateAffinePartial2D` on ORB/optical-flow features of the static
    background) and express keypoints in a stabilised frame before computing
    velocities. This is the single biggest accuracy win for GCT, because pans
    contaminate vertical velocity.
  - **Calibration:** pixel→cm from the athlete's standing height
    (`pixel_scale_cm`) is the cheap option; better is a known scene scale (track
    lane width = 1.22 m, hurdle/marking spacing) via a one-time homography.

## 4. Optimization (speed vs accuracy on Colab)

You're right that `yolov8n-pose` at 480p trades away the precision gait needs.
Priorities:

1. **Move up the model, down the resolution.** `yolov8s-pose` or `yol11s-pose`
   at `imgsz=480` is usually a better accuracy/speed point than `n` at 640.
   Gait needs joint *precision*, which the `n` model gives up first.
2. **fp16 (`half=True`)** — implemented; ~2x on a T4 for free.
3. **TensorRT export:** `YOLO(model).export(format='engine', half=True)` then
   load the `.engine`. Typically 2–4x over PyTorch on the same GPU — the highest
   single-step speedup, do this before sacrificing model size.
4. **ROI tracking:** once your athlete's track is established, crop a padded box
   around her and run pose on the crop at higher effective resolution. More
   precision *and* less compute than full-frame.
5. **`vid_stride` carefully:** stride 2 halves work but halves temporal
   resolution — bad for GCT (already short). Prefer TensorRT + ROI over striding
   for gait. Implemented and timing-correct, but use sparingly.
6. **Tracker tuning:** ByteTrack with raised `track_buffer` keeps IDs through
   brief occlusions, reducing fragmentation (and manual re-picking).

Rule of thumb: get speed from **TensorRT + ROI crop + fp16**, keep accuracy by
**not** dropping below `s`-class at `imgsz≥480` for the joints that drive gait.

## 5. Data schema for longitudinal profiling

*Implemented:* `features.FeatureRecord` + JSONL store (`append_record`,
`load_records`, `record_from_metrics`). One analysis → one record:

```
{ athlete, race, event, date, race_stage, gear, fatigue_onset,
  ground_contact_ms, cadence_spm, vertical_osc_cm, head_sway_cm, hip_rom_deg,
  economy_gain_pct, predicted_time_s,
  source, model, fps, segments_seen, notes }
```

Why JSONL: append-only, diff-friendly, one row per (athlete, race, stage), so
you can chart a metric across a season or compare opening vs final-lap form
within one race (`race_stage`). Provenance fields (`source`, `model`, `fps`,
`segments_seen`) keep a profile auditable — you always know how solid each row
is. For scale, the same schema drops cleanly into a Parquet/DuckDB table later.

## 6. Recommended end-to-end architecture

```
video ─▶ [ROI-aware pose: yolo s-class, TensorRT, fp16]
      ─▶ per-person tracks (ByteTrack, high track_buffer)
      ─▶ camera-motion compensation (homography on background)
      ─▶ confidence gating ▶ gap interpolation ▶ One-Euro smoothing   (preprocess)
      ─▶ presence-based segmentation into work clips                  (split_on_gaps)
      ─▶ view gating (side-on only for sagittal metrics)
      ─▶ per-clip kinematics: GCT (velocity), hip ROM, vert. osc., sway, cadence
      ─▶ duration-weighted aggregate across clips                     (aggregate_segments)
      ─▶ economy % vs reference ▶ ⅔ time transfer                     (biomechanics/economy)
      ─▶ FeatureRecord ▶ JSONL longitudinal store                     (features)
```

Bracketed/named items in the right column are already in this repo; the rest
(TensorRT, ROI crop, homography stabilisation, One-Euro, OCR stage-labelling)
are the prioritised upgrades, roughly in order of accuracy payoff.
