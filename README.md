# dance-pose-tracker
A computer vision app using YOLOv9 to track dance poses and body movements frame-by-frame from YouTube URLs.

## Athletic Performance Predictor

A data-driven model (`athlete_predictor/`) that puts athletes from different
eras on equal footing, so you can answer questions like:

> *What would peak Kenenisa Bekele run for the marathon in current super shoes?*

### How it works

Every recorded performance is normalized before athletes are compared:

1. **Gear** — the shoe-technology advantage actually worn is stripped out
   (carbon-plate super shoes are worth roughly 1.4–2.3% on the road,
   super spikes ~0.8% on the track, per published running-economy studies).
2. **Age** — times are adjusted to the athlete's physical peak with a
   quadratic age curve (peak ≈ 25 on the track, ≈ 28 on the road).
3. **Distance** — the neutral peak time is projected to the target event
   with a Riegel power law. The endurance exponent is *fitted per athlete*
   from their own PBs when they have marks at two or more distances
   (a tiny log-log least-squares regression), with a small fatigue bump
   when crossing from the track to the full marathon.
4. The target gear advantage is re-applied, estimates from all of the
   athlete's performances are combined with a distance-weighted geometric
   mean, and parameter uncertainty is swept to produce a low–high range.

### Running economy (form efficiency)

Energy spent on motion that doesn't drive you forward — head wobble,
excessive vertical bounce, lateral sway, overstriding — is *wasted
running economy*. This is exactly what a pose tracker (the rest of this
repo) can measure frame-by-frame. The `--economy-gain` and `--fix`
options model what happens if an athlete cleans that up:

```bash
# A named form fault with a built-in ballpark cost
python -m athlete_predictor predict --athlete gidey --event 5000m \
    --gear super_spikes --fix head_wobble

# Or set the economy improvement directly (in %)
python -m athlete_predictor predict --athlete gidey --event 5000m \
    --gear super_spikes --economy-gain 1.5
```

Economy gains don't transfer 1:1 to time — a runner near VO₂max converts
only ~two-thirds of an oxygen saving into speed (`ECONOMY_TO_TIME`), so a
1.5% cleaner stride buys roughly 1% off the clock. Built-in fault costs
(`head_wobble`, `vertical_oscillation`, `overstriding`, `arm_crossover`)
are starting-point estimates you can tune in `economy.py`.

#### Fatigue-driven faults

A flaw that only appears late in the race (a head wobble that creeps in
as the athlete tires) costs less over the whole distance than one present
from the gun. `--fatigue-onset FRACTION` ramps the fault in linearly from
that fraction of race distance to the finish, so its *average* cost is
`gain × (1 − onset) / 2`:

```bash
# A 1.5% wobble that only bites from 60% distance onward
python -m athlete_predictor predict --athlete "Senayet Getachew" \
    --event 5000m --gear super_spikes --fix head_wobble --fatigue-onset 0.6
```

#### Form check from pose-tracker metrics (`formcheck`)

Ground contact time, vertical oscillation, cadence and head sway all come
straight out of side-on pose tracking — the "floating" stride is a short
ground contact with a high flight-to-contact ratio; a stiff, heavy stride
is a long contact. `formcheck` scores those measured numbers against an
elite reference (`biomechanics.py`) and reports the economy — and time —
on the table:

```bash
python -m athlete_predictor formcheck --ground-contact-ms 235 --cadence-spm 182 \
    --athlete "Freweyni Hailu" --event 5000m --gear super_spikes
```

The sensitivities are literature-based estimates; the *inputs* are meant
to be real measurements from the tracker, not guesses.

#### From a race video to a time (`video.py` + `analyze`)

You can feed those metrics straight from footage instead of typing them.
The flow splits the broadcast into camera shots, pose-tracks each shot,
lets you pick your athlete in each one, and averages her gait over only
the footage where she's actually visible — off-screen time inherits the
visible average:

```bash
pip install ultralytics opencv-python        # only needed for the video step
```

```python
from athlete_predictor.video import extract_segments, save_poses_json

shots = extract_segments("rome_5000m.mp4", model="yolov8n-pose.pt")
# track IDs reset at every camera cut, so point at your athlete per shot:
chosen = {0: 4, 2: 1, 5: 3}        # {shot_index: track_id}
save_poses_json("senayet_poses.json", shots, chosen,
                fps=shots.fps, athlete_height_cm=165)
```

```bash
# No CV dependencies needed from here on:
python -m athlete_predictor analyze --poses senayet_poses.json \
    --athlete "Senayet Getachew" --event 5000m --gear super_spikes \
    --fatigue-onset 0.6
```

The gait math (`pose_analysis.py`) is pure stdlib and fully tested;
`video.py` lazily imports the CV stack so the rest of the package works
without it. Pixel measurements are calibrated from the athlete's height
(`pixel_scale_cm`) and depend on camera angle, so the metrics are
estimates — good enough to compare athletes and run what-ifs, not lab
force-plate data.

#### One command on your own machine

If the video lives on your PC, the whole pipeline is a single command (the
pose step needs `pip install ultralytics opencv-python`):

```bash
# First pass: track everyone and write per-shot previews + the track IDs.
python -m athlete_predictor track --video rome_5000m.mp4 --previews previews/

# Look at previews/shot_*.jpg, then pick your athlete per shot and score her:
python -m athlete_predictor track --video rome_5000m.mp4 \
    --pick "0:1,2:3,5:2" --athlete-height-cm 165 \
    --athlete "Senayet Getachew" --event 5000m --gear super_spikes --fatigue-onset 0.6
```

`--auto` picks the longest-tracked runner per shot if you don't want to
choose by hand (it can grab the wrong person, so the previews are there to
check). The heavy pose step runs locally; the small `poses.json` it writes
can be analysed anywhere with `analyze`.

#### No GPU? Run it on Google Colab

`notebooks/colab_gait_analysis.ipynb` is a ready-to-run notebook: pick a
GPU runtime, upload your video, and it pose-tracks each camera shot,
shows you a labelled preview of every runner's track ID so you can point
at your athlete, then prints her gait metrics and equalized time. Open it
via [Colab](https://colab.research.google.com/github/henosss/dance-pose-tracker/blob/claude/athletic-performance-predictor-gmhkll/notebooks/colab_gait_analysis.ipynb).

The bundled dataset (`athlete_predictor/data/performances.csv`) contains
well-known career bests for Bekele, Kipchoge, Kiptum, Gebrselassie, Tergat,
Cheptegei, Farah, Radcliffe, Assefa, Chepngetich and Gidey, each tagged with
the shoe technology of its era. Add rows to compare more athletes.

### Usage

No dependencies — pure Python 3 standard library.

```bash
# Everything on record + fitted personal endurance exponents
python -m athlete_predictor list

# Peak Bekele, marathon, current super shoes (partial names are fine)
python -m athlete_predictor predict --athlete bekele --event marathon

# Equalized leaderboard: everyone at their peak, in the same shoes
python -m athlete_predictor compare --event marathon \
    --athletes bekele kipchoge kiptum gebrselassie

# Rewind instead: what would Kiptum run in pre-2017 flats?
python -m athlete_predictor predict --athlete kiptum --gear classic_flats
```

Example output:

```
Kenenisa Bekele at physical peak, marathon, gear: superfoam_2023
  predicted: 1:59:16
  range:     1:58:10 - 2:06:14
  based on:
    5000m            12:37.35 (2004, classic_spikes) -> 2:02:33
    10000m           26:17.53 (2005, classic_spikes) -> 2:01:06
    marathon          2:01:41 (2019, vaporfly) -> 1:58:39
```

Run the tests with:

```bash
python -m unittest discover -s tests
```

**Disclaimer:** the gear, age and distance adjustments are mid-range
estimates from public research, not exact physics — treat the outputs as
informed what-ifs, not certainties.
