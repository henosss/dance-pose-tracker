import unittest

from athlete_predictor import (
    compare,
    fit_personal_exponent,
    format_time,
    load_performances,
    neutral_peak_time,
    parse_time,
    predict,
)


class TestTimeParsing(unittest.TestCase):
    def test_parse_marathon(self):
        self.assertEqual(parse_time("2:01:41"), 2 * 3600 + 60 + 41)

    def test_parse_track(self):
        self.assertAlmostEqual(parse_time("26:17.53"), 26 * 60 + 17.53)

    def test_roundtrip(self):
        self.assertEqual(format_time(parse_time("2:01:41")), "2:01:41")
        self.assertEqual(format_time(parse_time("12:37.35")), "12:37.35")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_time("1:2:3:4")


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perfs = load_performances()

    def test_dataset_loads(self):
        self.assertGreaterEqual(len(self.perfs), 15)

    def test_neutralizing_super_shoes_slows_the_time(self):
        kiptum = next(p for p in self.perfs if p.athlete == "Kelvin Kiptum")
        self.assertGreater(neutral_peak_time(kiptum), kiptum.time_s)

    def test_bekele_track_exponent_is_plausible(self):
        bekele = [p for p in self.perfs if p.athlete == "Kenenisa Bekele"]
        b = fit_personal_exponent(bekele)
        self.assertGreater(b, 1.0)
        self.assertLess(b, 1.12)

    def test_single_distance_athlete_has_no_personal_exponent(self):
        kiptum = [p for p in self.perfs if p.athlete == "Kelvin Kiptum"]
        self.assertIsNone(fit_personal_exponent(kiptum))

    def test_peak_bekele_in_current_gear_beats_his_actual_marathon(self):
        pred = predict(self.perfs, "Kenenisa Bekele", "marathon", "superfoam_2023")
        actual = parse_time("2:01:41")
        self.assertLess(pred.time_s, actual)
        # Sanity bounds: faster than 2:03, slower than world-record fantasy land.
        self.assertGreater(pred.time_s, parse_time("1:55:00"))
        self.assertLess(pred.time_s, parse_time("2:03:00"))
        self.assertLess(pred.low_s, pred.time_s)
        self.assertGreater(pred.high_s, pred.time_s)

    def test_better_gear_means_faster_prediction(self):
        classic = predict(self.perfs, "Haile Gebrselassie", "marathon", "classic_flats")
        modern = predict(self.perfs, "Haile Gebrselassie", "marathon", "superfoam_2023")
        self.assertLess(modern.time_s, classic.time_s)

    def test_compare_is_sorted_and_equalized(self):
        names = ["Kenenisa Bekele", "Eliud Kipchoge", "Kelvin Kiptum", "Haile Gebrselassie"]
        preds = compare(self.perfs, names, "marathon", "superfoam_2023")
        times = [p.time_s for p in preds]
        self.assertEqual(times, sorted(times))
        self.assertEqual({p.athlete for p in preds}, set(names))

    def test_unknown_athlete_raises(self):
        with self.assertRaises(ValueError):
            predict(self.perfs, "Usain Bolt", "marathon", "superfoam_2023")

    def test_economy_gain_makes_athlete_faster(self):
        base = predict(self.perfs, "Letesenbet Gidey", "5000m", "super_spikes")
        cleaned = predict(
            self.perfs, "Letesenbet Gidey", "5000m", "super_spikes", economy_gain=2.0
        )
        self.assertLess(cleaned.time_s, base.time_s)

    def test_economy_gain_transfers_at_about_two_thirds(self):
        from athlete_predictor import ECONOMY_TO_TIME, time_factor

        base = predict(self.perfs, "Letesenbet Gidey", "5000m", "super_spikes")
        cleaned = predict(
            self.perfs, "Letesenbet Gidey", "5000m", "super_spikes", economy_gain=2.0
        )
        # A 2% economy gain should save ~2% * transfer of the clock.
        self.assertAlmostEqual(cleaned.time_s / base.time_s, time_factor(2.0), places=6)
        self.assertAlmostEqual(time_factor(2.0), 1.0 - 0.02 * ECONOMY_TO_TIME, places=9)

    def test_zero_economy_gain_is_a_no_op(self):
        base = predict(self.perfs, "Letesenbet Gidey", "5000m", "super_spikes")
        same = predict(
            self.perfs, "Letesenbet Gidey", "5000m", "super_spikes", economy_gain=0.0
        )
        self.assertEqual(base.time_s, same.time_s)

    def test_late_onset_fault_saves_less_than_whole_race_fault(self):
        whole = predict(self.perfs, "Senayet Getachew", "5000m", "super_spikes", 1.5)
        late = predict(
            self.perfs, "Senayet Getachew", "5000m", "super_spikes", 1.5,
            fatigue_onset=0.6,
        )
        base = predict(self.perfs, "Senayet Getachew", "5000m", "super_spikes")
        self.assertLess(whole.time_s, late.time_s)   # whole-race fix gains more
        self.assertLess(late.time_s, base.time_s)     # but a late fix still helps


class TestFatigueAndBiomechanics(unittest.TestCase):
    def test_fatigue_ramp_matches_formula(self):
        from athlete_predictor import fatigue_weighted_gain

        # ramp from 60% distance to finish: average cost = 1.5 * 0.4 / 2
        self.assertAlmostEqual(fatigue_weighted_gain(1.5, 0.6), 1.5 * 0.4 / 2)
        self.assertEqual(fatigue_weighted_gain(2.0, 1.0), 0.0)  # never bites

    def test_metric_at_reference_costs_nothing(self):
        from athlete_predictor import ELITE_REFERENCE, metric_penalty

        for name, ref in ELITE_REFERENCE.items():
            self.assertEqual(metric_penalty(name, ref), 0.0)

    def test_longer_ground_contact_costs_economy(self):
        from athlete_predictor import metric_penalty

        # 230 ms vs 180 ms reference -> 50 ms * 0.06 = 3.0%
        self.assertAlmostEqual(metric_penalty("ground_contact_ms", 230), 3.0)

    def test_better_than_reference_is_not_a_bonus(self):
        from athlete_predictor import metric_penalty

        self.assertEqual(metric_penalty("ground_contact_ms", 160), 0.0)
        self.assertEqual(metric_penalty("cadence_spm", 200), 0.0)

    def test_flight_to_contact_ratio(self):
        from athlete_predictor import flight_to_contact_ratio

        floaty = flight_to_contact_ratio(160, 140)
        heavy = flight_to_contact_ratio(240, 90)
        self.assertGreater(floaty, heavy)


def _synthetic_runner(n_frames=200, fps=50, foot_period=36, contact_frames=13,
                      bounce_px=5.0, sway_px=4.0, hip_y=300.0):
    """A clean synthetic gait signal with known cadence and contact time."""
    import math

    from athlete_predictor.pose_analysis import Sample

    samples = []
    for i in range(n_frames):
        com_y = hip_y + bounce_px * math.sin(2 * math.pi * i / (foot_period / 2))
        com_x = 100.0 + 3.0 * i  # moving forward
        nose_x = com_x + sway_px * math.sin(2 * math.pi * i / (foot_period / 2))
        nose_y = com_y - 60.0
        left_planted = (i % foot_period) < contact_frames
        right_planted = ((i + foot_period // 2) % foot_period) < contact_frames
        l_ankle_y = com_y + (90.0 if left_planted else 40.0)
        r_ankle_y = com_y + (90.0 if right_planted else 40.0)
        samples.append(Sample(t=i / fps, kp={
            "nose": (nose_x, nose_y),
            "left_hip": (com_x - 10, com_y), "right_hip": (com_x + 10, com_y),
            "left_ankle": (com_x - 10, l_ankle_y),
            "right_ankle": (com_x + 10, r_ankle_y),
        }))
    return samples, fps


class TestPoseAnalysis(unittest.TestCase):
    def setUp(self):
        self.samples, self.fps = _synthetic_runner()

    def test_cadence_and_contact_recovered(self):
        from athlete_predictor.pose_analysis import cadence_and_contact

        cadence, contact = cadence_and_contact(self.samples, self.fps)
        # foot_period 36 @ 50fps -> a strike every 0.36s -> ~166 spm
        self.assertTrue(150 < cadence < 185, cadence)
        # contact 13 frames @ 50 fps -> 260 ms
        self.assertTrue(230 < contact < 290, contact)

    def test_oscillation_and_sway_positive(self):
        from athlete_predictor.pose_analysis import (
            head_sway_cm, pixel_scale_cm, vertical_oscillation_cm,
        )

        scale = pixel_scale_cm(self.samples, 165)
        self.assertGreater(scale, 0)
        self.assertGreater(vertical_oscillation_cm(self.samples, scale), 0)
        self.assertGreater(head_sway_cm(self.samples, scale), 0)

    def test_aggregate_uses_average_when_metric_missing(self):
        from athlete_predictor.pose_analysis import aggregate_segments

        seg_full = {"cadence_spm": 180, "ground_contact_ms": 200,
                    "vertical_osc_cm": 8, "head_sway_cm": 3}
        seg_no_head = {"cadence_spm": 170, "ground_contact_ms": 220,
                       "vertical_osc_cm": 9, "head_sway_cm": None}
        out = aggregate_segments([seg_full, seg_no_head], weights=[1, 1])
        self.assertAlmostEqual(out["cadence_spm"], 175)        # averaged
        self.assertAlmostEqual(out["head_sway_cm"], 3)         # only the seen one

    def test_split_on_gaps(self):
        from athlete_predictor.pose_analysis import Sample, split_on_gaps

        samples = [Sample(t, {}) for t in (0.0, 0.02, 0.04, 2.0, 2.02)]
        clips = split_on_gaps(samples, max_gap_s=0.4)
        self.assertEqual([len(c) for c in clips], [3, 2])

    def test_hip_rom_recovered(self):
        import math

        from athlete_predictor.pose_analysis import Sample, hip_rom_deg

        samples = []
        for i in range(20):
            dx = 30 * math.sin(2 * math.pi * i / 10)   # thigh swings +/- forward
            samples.append(Sample(t=i / 50, kp={
                "left_hip": (100.0, 100.0), "left_knee": (100.0 + dx, 180.0),
            }))
        rom = hip_rom_deg(samples)
        # atan2(30, 80) ~ 20.6 deg each way -> ~41 deg peak-to-peak
        self.assertTrue(35 < rom < 47, rom)

    def test_preprocess_interpolates_short_gaps(self):
        from athlete_predictor.pose_analysis import Sample, preprocess

        samples = [
            Sample(0.00, {"nose": (0.0, 0.0)}),
            Sample(0.02, {}),                       # occluded frame
            Sample(0.04, {"nose": (4.0, 0.0)}),
        ]
        out = preprocess(samples, max_gap_s=0.3, smooth_window=1)
        self.assertIn("nose", out[1].kp)             # gap was filled
        self.assertAlmostEqual(out[1].kp["nose"][0], 2.0)  # linear midpoint

    def test_feature_record_roundtrip(self):
        import json
        import tempfile

        from athlete_predictor.features import (
            FeatureRecord, append_record, load_records, record_from_metrics,
        )

        rec = record_from_metrics(
            "Senayet Getachew",
            {"cadence_spm": 200, "hip_rom_deg": 38, "unknown_key": 1},
            race="Rome 2026", gear="super_spikes", race_stage="final_laps",
        )
        self.assertEqual(rec.cadence_spm, 200)
        self.assertNotIn("unknown_key", rec.to_dict())   # filtered

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        append_record(path, rec)
        append_record(path, FeatureRecord(athlete="Freweyni Hailu", cadence_spm=190))
        loaded = load_records(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].race, "Rome 2026")

    def test_parse_pick(self):
        from athlete_predictor.cli import parse_pick

        self.assertEqual(parse_pick("0:1,2:3"), {0: 1, 2: 3})
        self.assertEqual(parse_pick(" 5:2 "), {5: 2})
        self.assertEqual(parse_pick(""), {})

    def test_analyze_command_end_to_end(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        from athlete_predictor.cli import main

        samples, fps = _synthetic_runner()
        doc = {"fps": fps, "athlete_height_cm": 165, "segments": [[
            {"t": s.t, "kp": {k: list(v) for k, v in s.kp.items()}} for s in samples
        ]]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            path = f.name

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["analyze", "--poses", path, "--athlete", "Senayet Getachew",
                  "--event", "5000m", "--gear", "super_spikes"])
        out = buf.getvalue()
        self.assertIn("running-economy gain available", out)
        self.assertIn("ground_contact_ms", out)


if __name__ == "__main__":
    unittest.main()
