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
