"""A storage schema for longitudinal athlete-profiling.

One extracted analysis -> one ``FeatureRecord``: the measured gait metrics
plus the metadata you profile against (race, stage, gear, fatigue model,
provenance). Records append to a JSON Lines file so you can accumulate many
races per athlete over time and diff them.
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class FeatureRecord:
    # --- identity / metadata ---
    athlete: str
    race: str = ""               # e.g. "Rome Diamond League 2026"
    event: str = "5000m"
    date: str = ""               # ISO date of the race
    race_stage: str = "full"     # "full" | "opening" | "final_laps" | "kick" ...
    gear: str = ""               # shoe tech actually worn
    fatigue_onset: float = None  # fraction of race the modelled fault ramps in

    # --- measured gait (race-average, after cleanup) ---
    ground_contact_ms: float = None
    cadence_spm: float = None
    vertical_osc_cm: float = None
    head_sway_cm: float = None
    hip_rom_deg: float = None

    # --- derived ---
    economy_gain_pct: float = None   # economy available vs elite reference
    predicted_time_s: float = None   # equalized/what-if time, if computed

    # --- provenance (so a profile is auditable) ---
    source: str = ""             # video filename / URL
    model: str = ""              # pose model used
    fps: float = None
    segments_seen: int = None    # how many visible clips fed the average
    notes: str = ""

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


def append_record(path, record):
    """Append one FeatureRecord to a JSONL store."""
    with open(path, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def load_records(path):
    """Load all FeatureRecords from a JSONL store."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(FeatureRecord(**json.loads(line)))
    return records


def record_from_metrics(athlete, metrics, **metadata):
    """Build a FeatureRecord from an aggregated metrics dict plus metadata."""
    fields = {f for f in FeatureRecord.__dataclass_fields__}
    known = {k: v for k, v in {**metrics, **metadata}.items() if k in fields}
    return FeatureRecord(athlete=athlete, **known)
