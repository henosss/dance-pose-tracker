from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    name: str
    distance_m: float
    surface: str  # "track" or "road"


EVENTS = {
    "5000m": Event("5000m", 5000.0, "track"),
    "10000m": Event("10000m", 10000.0, "track"),
    "half_marathon": Event("half_marathon", 21097.5, "road"),
    "marathon": Event("marathon", 42195.0, "road"),
}


@dataclass(frozen=True)
class Performance:
    athlete: str
    sex: str
    birth_year: int
    event: str
    time_s: float
    year: int
    venue: str
    shoe_tech: str

    @property
    def distance_m(self) -> float:
        return EVENTS[self.event].distance_m

    @property
    def surface(self) -> str:
        return EVENTS[self.event].surface

    @property
    def age(self) -> int:
        return self.year - self.birth_year


@dataclass
class Prediction:
    athlete: str
    event: str
    gear: str
    time_s: float
    low_s: float
    high_s: float
    basis: list = field(default_factory=list)  # (source Performance, estimate_s)
