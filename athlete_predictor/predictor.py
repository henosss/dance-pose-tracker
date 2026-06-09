"""Core prediction model.

Pipeline for every recorded performance:

1. Strip the shoe-technology advantage of the gear actually worn
   (``time / (1 - benefit)``) to get a "neutral gear" time.
2. Strip the age handicap relative to the athlete's physical peak
   (quadratic curve around the typical peak age for the surface).
3. Project the neutral peak time to the target distance with a Riegel
   power law. The exponent is fitted per athlete from their own PBs
   when they have marks at two or more distances; otherwise a default
   elite exponent is used. Projections that cross from the track to
   the full marathon get a small fatigue/fueling bump.
4. Re-apply the *target* gear advantage.

Estimates from several source performances are combined with a
distance-weighted geometric mean, and a low/high band is produced by
sweeping the exponent and gear benefit through their uncertainty.
"""

import math

from .gear import GEAR_UNCERTAINTY, SHOE_BENEFIT
from .models import EVENTS, Prediction

DEFAULT_EXPONENT = 1.06     # classic Riegel exponent for elite endurance runners
MARATHON_BUMP = 0.015       # extra fatigue/fueling cost when crossing track -> marathon
EXPONENT_UNCERTAINTY = 0.012

PEAK_AGE = {"track": 25, "road": 28}
AGE_CURVE = {"track": 0.0003, "road": 0.0002}  # quadratic time penalty per year^2 off peak
MAX_AGE_PENALTY = 0.06


def parse_time(text):
    """Parse 'H:MM:SS', 'M:SS.ss' or plain seconds into seconds."""
    parts = text.strip().split(":")
    if len(parts) > 3:
        raise ValueError(f"unparseable time: {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def format_time(seconds):
    if seconds >= 3600:
        h, rem = divmod(round(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}"
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def age_factor(age, surface):
    """Multiplier on time relative to the athlete's peak (>= 1.0)."""
    penalty = AGE_CURVE[surface] * (age - PEAK_AGE[surface]) ** 2
    return 1.0 + min(penalty, MAX_AGE_PENALTY)


def neutral_peak_time(perf, gear_delta=0.0):
    """Time stripped of shoe tech and adjusted to the athlete's peak age."""
    benefit = SHOE_BENEFIT[perf.shoe_tech]
    if benefit > 0:
        benefit += gear_delta
    return perf.time_s / (1.0 - benefit) / age_factor(perf.age, perf.surface)


def fit_personal_exponent(perfs):
    """Least-squares fit of log(time) vs log(distance) over an athlete's PBs.

    Returns None when fewer than two distinct distances are available.
    """
    best = {}
    for p in perfs:
        t = neutral_peak_time(p)
        if p.distance_m not in best or t < best[p.distance_m]:
            best[p.distance_m] = t
    if len(best) < 2:
        return None
    xs = [math.log(d) for d in best]
    ys = [math.log(t) for t in best.values()]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _exponent_for(source_d, target_d, personal_b):
    b = personal_b if personal_b is not None else DEFAULT_EXPONENT
    crosses_marathon = (
        min(source_d, target_d) <= EVENTS["10000m"].distance_m
        and max(source_d, target_d) >= EVENTS["marathon"].distance_m
    )
    if crosses_marathon:
        b += MARATHON_BUMP
    return b


def _estimate(perf, target, target_benefit, personal_b, b_delta=0.0, gear_delta=0.0):
    base = neutral_peak_time(perf)
    ratio = target.distance_m / perf.distance_m
    b = _exponent_for(perf.distance_m, target.distance_m, personal_b) + b_delta
    benefit = target_benefit + gear_delta if target_benefit > 0 else target_benefit
    return base * ratio**b * (1.0 - benefit)


def predict(performances, athlete, event, gear):
    """Predict `athlete`'s peak-form time for `event` wearing `gear`."""
    target = EVENTS[event]
    target_benefit = SHOE_BENEFIT[gear]
    perfs = [p for p in performances if p.athlete == athlete]
    if not perfs:
        raise ValueError(f"no performances on record for {athlete!r}")
    personal_b = fit_personal_exponent(perfs)

    basis, weights = [], []
    candidates = []
    for p in perfs:
        mid = _estimate(p, target, target_benefit, personal_b)
        ratio_gap = abs(math.log(target.distance_m / p.distance_m))
        weight = 1.0 / (0.25 + ratio_gap)
        basis.append((p, mid))
        weights.append(weight)
        for b_delta in (-EXPONENT_UNCERTAINTY, 0.0, EXPONENT_UNCERTAINTY):
            for gear_delta in (-GEAR_UNCERTAINTY, 0.0, GEAR_UNCERTAINTY):
                candidates.append(
                    _estimate(p, target, target_benefit, personal_b, b_delta, gear_delta)
                )

    log_mid = sum(w * math.log(t) for (_, t), w in zip(basis, weights)) / sum(weights)
    return Prediction(
        athlete=athlete,
        event=event,
        gear=gear,
        time_s=math.exp(log_mid),
        low_s=min(candidates),
        high_s=max(candidates),
        basis=basis,
    )


def compare(performances, athletes, event, gear):
    """Equalized leaderboard: every athlete at peak form in the same gear."""
    predictions = [predict(performances, a, event, gear) for a in athletes]
    predictions.sort(key=lambda p: p.time_s)
    return predictions
