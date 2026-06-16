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
from .pose_analysis import (
    Sample,
    aggregate_segments,
    hip_rom_deg,
    metrics_for_segment,
    preprocess,
    split_on_gaps,
)
from .features import FeatureRecord, append_record, load_records, record_from_metrics
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
    "Sample",
    "aggregate_segments",
    "hip_rom_deg",
    "metrics_for_segment",
    "preprocess",
    "split_on_gaps",
    "FeatureRecord",
    "append_record",
    "load_records",
    "record_from_metrics",
    "load_performances",
    "age_factor",
    "compare",
    "fit_personal_exponent",
    "format_time",
    "neutral_peak_time",
    "parse_time",
    "predict",
]
