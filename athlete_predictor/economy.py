"""Running-economy adjustment for biomechanical form.

Running economy is how much energy (oxygen) an athlete burns to hold a
given pace. Energy spent on motion that does not drive you forward --
head wobble, excessive vertical bounce, lateral arm/torso sway -- is
wasted economy. A pose tracker (the rest of this repo) is exactly the
tool that surfaces those flaws frame-by-frame.

If an athlete cleans up that wasted motion, their economy improves by
some percentage. Economy gains do NOT translate one-for-one into time:
a runner near VO2max only converts part of an oxygen saving into speed.
Published work (Hoogkamer 2018; Kipp, Kram & Hoogkamer 2019) puts the
transfer for super-shoe-scale economy gains at roughly two-thirds, so a
2% economy gain buys on the order of ~1.3% off the clock.

These are mid-range estimates, not exact physiology -- treat the output
as an informed what-if.
"""

# Fraction of a running-economy % gain that shows up as a time improvement.
ECONOMY_TO_TIME = 0.65
ECONOMY_TO_TIME_UNCERTAINTY = 0.15

# Rough economy cost (in % of running economy) attributed to common,
# visually obvious form faults a pose tracker can flag. Ballpark figures
# meant as starting points for "what if they fixed this" questions.
FORM_FAULT_COST = {
    "head_wobble": 1.5,            # bobbing / side-to-side head motion
    "vertical_oscillation": 2.0,  # excessive up-and-down bounce
    "overstriding": 1.5,          # braking on heel strike ahead of center of mass
    "arm_crossover": 1.0,         # arms swinging across the midline
}


def time_factor(economy_gain_pct, transfer=ECONOMY_TO_TIME):
    """Multiplier on race time for a running-economy improvement.

    `economy_gain_pct` is the percentage reduction in energy cost
    (e.g. 2.0 for a 2% better economy). Returns a value <= 1.0.
    """
    return 1.0 - (economy_gain_pct / 100.0) * transfer


def fatigue_weighted_gain(gain_pct, onset):
    """Economy gain available when a fault only shows up as fatigue sets in.

    A fault that creeps in late (a head wobble that appears once the
    athlete tires) costs less over the whole race than one present from
    the gun. We model it ramping linearly from no effect at `onset`
    (fraction of race distance, 0-1) to its full cost at the finish, so
    its average cost over the race is ``gain * (1 - onset) / 2``.

    `onset` of 1.0 means the fault never really bites (no cost); 0.0
    means it builds gradually across the entire race.
    """
    if not 0.0 <= onset <= 1.0:
        raise ValueError("onset must be between 0 and 1")
    return gain_pct * (1.0 - onset) / 2.0

