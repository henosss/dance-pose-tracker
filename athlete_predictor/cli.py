import argparse
import sys

from .data import load_performances
from .gear import CURRENT_GEAR, SHOE_BENEFIT
from .models import EVENTS
from .predictor import compare, fit_personal_exponent, format_time, predict


def _resolve_gear(gear, event):
    if gear == "current":
        return CURRENT_GEAR[EVENTS[event].surface]
    if gear not in SHOE_BENEFIT:
        sys.exit(f"unknown gear {gear!r}; choices: current, {', '.join(SHOE_BENEFIT)}")
    return gear


def _match_athlete(performances, name):
    names = sorted({p.athlete for p in performances})
    exact = [n for n in names if n.lower() == name.lower()]
    if exact:
        return exact[0]
    partial = [n for n in names if name.lower() in n.lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        sys.exit(f"ambiguous athlete {name!r}: matches {', '.join(partial)}")
    sys.exit(f"unknown athlete {name!r}; run `list` to see who is on record")


def cmd_list(performances, _args):
    print(f"{'athlete':22} {'event':14} {'time':>10} {'year':>5}  gear")
    for p in sorted(performances, key=lambda p: (p.athlete, p.distance_m)):
        print(f"{p.athlete:22} {p.event:14} {format_time(p.time_s):>10} {p.year:>5}  {p.shoe_tech}")
    print()
    print("fitted personal endurance exponents (Riegel):")
    for name in sorted({p.athlete for p in performances}):
        b = fit_personal_exponent([p for p in performances if p.athlete == name])
        print(f"  {name:22} {f'{b:.4f}' if b else '(default, single distance)'}")


def cmd_predict(performances, args):
    gear = _resolve_gear(args.gear, args.event)
    athlete = _match_athlete(performances, args.athlete)
    pred = predict(performances, athlete, args.event, gear)
    print(f"{athlete} at physical peak, {args.event.replace('_', ' ')}, gear: {gear}")
    print(f"  predicted: {format_time(pred.time_s)}")
    print(f"  range:     {format_time(pred.low_s)} - {format_time(pred.high_s)}")
    print("  based on:")
    for perf, est in pred.basis:
        print(
            f"    {perf.event:14} {format_time(perf.time_s):>10} ({perf.year}, "
            f"{perf.shoe_tech}) -> {format_time(est)}"
        )


def cmd_compare(performances, args):
    gear = _resolve_gear(args.gear, args.event)
    athletes = [_match_athlete(performances, a) for a in args.athletes]
    preds = compare(performances, athletes, args.event, gear)
    print(f"Equalized {args.event.replace('_', ' ')} - everyone at peak in {gear}:")
    leader = preds[0].time_s
    for i, p in enumerate(preds, 1):
        gap = p.time_s - leader
        gap_str = "  leader" if gap == 0 else f"  +{gap:.0f}s"
        print(
            f"  {i}. {p.athlete:22} {format_time(p.time_s)} "
            f"({format_time(p.low_s)} - {format_time(p.high_s)}){gap_str}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="athlete_predictor",
        description="Compare athletes on equal terms: peak form, same shoe tech.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show all performances and fitted exponents")

    p = sub.add_parser("predict", help="predict one athlete's equalized time")
    p.add_argument("--athlete", required=True)
    p.add_argument("--event", choices=sorted(EVENTS), default="marathon")
    p.add_argument("--gear", default="current", help="shoe tech (default: current)")

    c = sub.add_parser("compare", help="equalized leaderboard for several athletes")
    c.add_argument("--athletes", nargs="+", required=True)
    c.add_argument("--event", choices=sorted(EVENTS), default="marathon")
    c.add_argument("--gear", default="current", help="shoe tech (default: current)")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    performances = load_performances()
    {"list": cmd_list, "predict": cmd_predict, "compare": cmd_compare}[args.command](
        performances, args
    )


if __name__ == "__main__":
    main()
