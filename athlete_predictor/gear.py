"""Shoe technology adjustment factors.

Values are the approximate fraction of finish time saved versus
pre-2017 racing shoes, taken from the middle of published running
economy studies (Hoogkamer 2018, Barnes & Kilding 2019) and
statistical analyses of race results (Bermon 2021, Guinness 2023).
They are estimates, not exact physics.
"""

SHOE_BENEFIT = {
    # Road racing shoes
    "classic_flats": 0.000,     # pre-2017 racing flats (baseline)
    "vaporfly": 0.014,          # first-gen carbon plate + PEBA foam, 2017-2019
    "alphafly": 0.019,          # second-gen super shoes, 2020-2022
    "superfoam_2023": 0.023,    # current generation, 2023+
    # Track spikes
    "classic_spikes": 0.000,    # pre-2020 spikes (baseline)
    "super_spikes": 0.008,      # carbon/foam "super spikes", 2020+
}

# How far the true benefit could plausibly differ from the table above.
GEAR_UNCERTAINTY = 0.004

# What "current gear" means per surface.
CURRENT_GEAR = {"road": "superfoam_2023", "track": "super_spikes"}
