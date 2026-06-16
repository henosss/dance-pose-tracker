"""From pose-tracker metrics to a running-economy estimate.

This is the bridge between the pose tracker (which measures how an
athlete moves) and the performance predictor (which turns economy into
time). All of these inputs are things you can pull straight out of a
side-on video with pose tracking:

* ground_contact_ms   -- time the foot is planted (foot keypoint stationary)
* flight_ms           -- time both feet are off the ground
* vertical_osc_cm     -- up/down travel of the hips / center of mass
* cadence_spm         -- steps per minute
* head_sway_cm        -- side-to-side travel of the head keypoint

The "felt like she was in the air" stride is a high flight-to-contact
ratio with a short ground contact; a stiff, heavy stride is a long
contact time. We score each metric against an elite distance-running
reference and add up the running-economy cost of being worse than it.

The sensitivities below are mid-range estimates drawn from running-economy
literature (e.g. Santos-Concejero on ground contact, Folland 2017 on
vertical oscillation and stride mechanics). They are starting points to
be tuned against real measurements, not settled constants.
"""

# Elite distance-running reference values (good, economical form).
ELITE_REFERENCE = {
    "ground_contact_ms": 180.0,
    "vertical_osc_cm": 7.0,
    "cadence_spm": 190.0,
    "head_sway_cm": 2.0,
}

# Running-economy cost in % per unit *worse* than the reference.
# "Worse" means longer contact, more bounce, more head sway, lower cadence.
SENSITIVITY = {
    "ground_contact_ms": 0.06,   # per ms longer than reference
    "vertical_osc_cm": 0.80,     # per cm more bounce than reference
    "cadence_spm": 0.06,         # per spm below reference (overstriding)
    "head_sway_cm": 0.50,        # per cm more side-to-side head travel
}

# Metrics where a higher value is worse (vs lower-is-worse for cadence).
_HIGHER_IS_WORSE = {"ground_contact_ms", "vertical_osc_cm", "head_sway_cm"}


def metric_penalty(name, value):
    """Running-economy % cost of one metric being worse than the reference.

    Returns 0 when the athlete is at or better than the reference.
    """
    if name not in SENSITIVITY:
        raise ValueError(f"unknown biomechanic metric {name!r}")
    ref = ELITE_REFERENCE[name]
    deficit = (value - ref) if name in _HIGHER_IS_WORSE else (ref - value)
    return max(0.0, deficit) * SENSITIVITY[name]


def economy_penalty(metrics):
    """Total running-economy % gain available by cleaning up to reference.

    `metrics` is a dict of any subset of the known metric names. The
    result is the economy improvement an athlete would gain by bringing
    every supplied metric to the elite reference -- feed it straight into
    ``predict(..., economy_gain=...)``.
    """
    return sum(metric_penalty(name, value) for name, value in metrics.items())


def flight_to_contact_ratio(ground_contact_ms, flight_ms):
    """How 'airborne' a stride is. Higher = floatier, like the winner."""
    if ground_contact_ms <= 0:
        raise ValueError("ground_contact_ms must be positive")
    return flight_ms / ground_contact_ms
