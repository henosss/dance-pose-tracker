import csv
from pathlib import Path

from .models import Performance
from .predictor import parse_time

DATA_FILE = Path(__file__).parent / "data" / "performances.csv"


def load_performances(path=None):
    path = Path(path) if path else DATA_FILE
    performances = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            performances.append(
                Performance(
                    athlete=row["athlete"],
                    sex=row["sex"],
                    birth_year=int(row["birth_year"]),
                    event=row["event"],
                    time_s=parse_time(row["time"]),
                    year=int(row["year"]),
                    venue=row["venue"],
                    shoe_tech=row["shoe_tech"],
                )
            )
    return performances
