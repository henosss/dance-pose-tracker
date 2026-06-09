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
