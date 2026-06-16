import argparse
import json
import sys

from .biomechanics import ELITE_REFERENCE, SENSITIVITY, economy_penalty, metric_penalty
from .data import load_performances
from .economy import FORM_FAULT_COST, fatigue_weighted_gain
from .gear import CURRENT_GEAR, SHOE_BENEFIT
from .models import EVENTS
from .pose_analysis import aggregate_segments, metrics_for_segment, segments_from_json
from .predictor import compare, fit_personal_exponent, format_time, predict


def _economy_gain(args):
    """Total running-economy % gain from --economy-gain plus any --fix faults."""
    gain = args.economy_gain
    for fault in getattr(args, "fix", None) or []:
        if fault not in FORM_FAULT_COST:
            import sys
            sys.exit(f"unknown form fault {fault!r}; choices: {', '.join(FORM_FAULT_COST)}")
        gain += FORM_FAULT_COST[fault]
    return gain


def _add_economy_args(parser):
    parser.add_argument(
        "--economy-gain", type=float, default=0.0,
        help="running-economy improvement in %% (e.g. 2 for 2%% cleaner form)",
    )
    parser.add_argument(
        "--fix", nargs="+", default=[], metavar="FAULT",
        help=f"form faults to clean up; choices: {', '.join(FORM_FAULT_COST)}",
    )
    parser.add_argument(
        "--fatigue-onset", type=float, default=None, metavar="FRACTION",
        help="treat the gain as a fatigue fault that only ramps in after this "
             "fraction of the race (e.g. 0.6 for a wobble that starts late)",
    )


def _effective_gain(args):
    """Economy gain actually applied, after any fatigue weighting."""
    gain = _economy_gain(args)
    if getattr(args, "fatigue_onset", None) is not None and gain:
        return fatigue_weighted_gain(gain, args.fatigue_onset)
    return gain


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
    raw_gain = _economy_gain(args)
    gain = _effective_gain(args)
    pred = predict(
        performances, athlete, args.event, gear, raw_gain, args.fatigue_onset
    )
    print(f"{athlete} at physical peak, {args.event.replace('_', ' ')}, gear: {gear}")
    if gain:
        baseline = predict(performances, athlete, args.event, gear)
        saved = baseline.time_s - pred.time_s
        note = ""
        if args.fatigue_onset is not None and raw_gain:
            note = (f"; {raw_gain:.1f}% fault from {args.fatigue_onset:.0%} race "
                    f"distance averages to {gain:.2f}%")
        print(
            f"  with +{gain:.2f}% effective running economy "
            f"(saves {saved:.1f}s vs current form{note})"
        )
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
    preds = compare(
        performances, athletes, args.event, gear,
        _economy_gain(args), args.fatigue_onset,
    )
    print(f"Equalized {args.event.replace('_', ' ')} - everyone at peak in {gear}:")
    leader = preds[0].time_s
    for i, p in enumerate(preds, 1):
        gap = p.time_s - leader
        gap_str = "  leader" if gap == 0 else f"  +{gap:.0f}s"
        print(
            f"  {i}. {p.athlete:22} {format_time(p.time_s)} "
            f"({format_time(p.low_s)} - {format_time(p.high_s)}){gap_str}"
        )


def cmd_formcheck(performances, args):
    metrics = {
        "ground_contact_ms": args.ground_contact_ms,
        "vertical_osc_cm": args.vertical_osc_cm,
        "cadence_spm": args.cadence_spm,
        "head_sway_cm": args.head_sway_cm,
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}
    if not metrics:
        sys.exit("give at least one measured metric, e.g. --ground-contact-ms 230")

    print("Form check vs elite reference (economy cost of being worse):")
    for name, value in metrics.items():
        ref = ELITE_REFERENCE[name]
        cost = metric_penalty(name, value)
        print(f"  {name:20} {value:>7.1f} (ref {ref:>6.1f})  -> +{cost:.2f}% economy")
    gain = economy_penalty(metrics)
    print(f"  total running-economy gain available: {gain:.2f}%")

    if args.athlete:
        athlete = _match_athlete(performances, args.athlete)
        gear = _resolve_gear(args.gear, args.event)
        base = predict(performances, athlete, args.event, gear)
        fixed = predict(performances, athlete, args.event, gear, gain)
        saved = base.time_s - fixed.time_s
        print(
            f"\nFor {athlete} at {args.event.replace('_', ' ')} ({gear}): "
            f"{format_time(base.time_s)} -> {format_time(fixed.time_s)} "
            f"(saves {saved:.1f}s if cleaned up to reference)"
        )


def cmd_analyze(performances, args):
    with open(args.poses) as f:
        segments, fps, height_cm = segments_from_json(json.load(f))
    per_segment = [metrics_for_segment(seg, fps, height_cm) for seg in segments]
    weights = [seg[-1].t - seg[0].t if len(seg) > 1 else 1.0 for seg in segments]
    metrics = aggregate_segments(per_segment, weights)

    print(f"Tracked {len(segments)} visible segment(s) at {fps:.0f} fps.")
    print("Race-average gait (off-screen time inherits the visible average):")
    measured = {}
    for name in ("ground_contact_ms", "cadence_spm", "vertical_osc_cm", "head_sway_cm"):
        value = metrics.get(name)
        if value is None:
            print(f"  {name:20} (not measurable from these segments)")
            continue
        measured[name] = value
        cost = metric_penalty(name, value)
        print(f"  {name:20} {value:>7.1f} (ref {ELITE_REFERENCE[name]:>6.1f})  -> +{cost:.2f}%")

    gain = economy_penalty(measured)
    print(f"  total running-economy gain available: {gain:.2f}%")

    if args.athlete and gain:
        athlete = _match_athlete(performances, args.athlete)
        gear = _resolve_gear(args.gear, args.event)
        base = predict(performances, athlete, args.event, gear)
        fixed = predict(performances, athlete, args.event, gear, gain, args.fatigue_onset)
        saved = base.time_s - fixed.time_s
        print(
            f"\n{athlete} at {args.event.replace('_', ' ')} ({gear}): "
            f"{format_time(base.time_s)} -> {format_time(fixed.time_s)} "
            f"if cleaned up to reference (saves {saved:.1f}s)"
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
    _add_economy_args(p)

    c = sub.add_parser("compare", help="equalized leaderboard for several athletes")
    c.add_argument("--athletes", nargs="+", required=True)
    c.add_argument("--event", choices=sorted(EVENTS), default="marathon")
    c.add_argument("--gear", default="current", help="shoe tech (default: current)")
    _add_economy_args(c)

    f = sub.add_parser(
        "formcheck",
        help="estimate economy cost of measured biomechanics (pose-tracker metrics)",
    )
    f.add_argument("--ground-contact-ms", type=float, default=None)
    f.add_argument("--vertical-osc-cm", type=float, default=None)
    f.add_argument("--cadence-spm", type=float, default=None)
    f.add_argument("--head-sway-cm", type=float, default=None)
    f.add_argument("--athlete", default=None, help="optional: show the time impact")
    f.add_argument("--event", choices=sorted(EVENTS), default="5000m")
    f.add_argument("--gear", default="current", help="shoe tech (default: current)")

    a = sub.add_parser(
        "analyze",
        help="read tracked pose segments (JSON from video.py) and score gait",
    )
    a.add_argument("--poses", required=True, help="poses JSON produced by video.py")
    a.add_argument("--athlete", default=None, help="optional: show the time impact")
    a.add_argument("--event", choices=sorted(EVENTS), default="5000m")
    a.add_argument("--gear", default="current", help="shoe tech (default: current)")
    a.add_argument("--fatigue-onset", type=float, default=None, metavar="FRACTION")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    performances = load_performances()
    {
        "list": cmd_list,
        "predict": cmd_predict,
        "compare": cmd_compare,
        "formcheck": cmd_formcheck,
        "analyze": cmd_analyze,
    }[args.command](performances, args)


if __name__ == "__main__":
    main()
