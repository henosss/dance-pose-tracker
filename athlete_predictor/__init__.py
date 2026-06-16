"""Athletic performance predictor.

Compares athletes on a level playing field by normalizing recorded
performances for shoe technology, age at the time of the race, and then
projecting across distances. Answers questions like: "What would peak
Kenenisa Bekele run for the marathon in current super shoes?"
"""

from .models import Event, Performance, Prediction, EVENTS
from .gear import SHOE_BENEFIT
from .economy import ECONOMY_TO_TIME, FORM_FAULT_COST, fatigue_weighted_gain, time_factor
from .biomechanics import (
    ELITE_REFERENCE,
    economy_penalty,
    flight_to_contact_ratio,
    metric_penalty,
)
from .data import load_performances
from .predictor import (
    age_factor,
    compare,
    fit_personal_exponent,
    format_time,
    neutral_peak_time,
    parse_time,
    predict,
)

__all__ = [
    "Event",
    "Performance",
    "Prediction",
    "EVENTS",
    "SHOE_BENEFIT",
    "ECONOMY_TO_TIME",
    "FORM_FAULT_COST",
    "fatigue_weighted_gain",
    "time_factor",
    "ELITE_REFERENCE",
    "economy_penalty",
    "flight_to_contact_ratio",
    "metric_penalty",
    "load_performances",
    "age_factor",
    "compare",
    "fit_personal_exponent",
    "format_time",
    "neutral_peak_time",
    "parse_time",
    "predict",
]
