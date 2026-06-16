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


if __name__ == "__main__":
    unittest.main()
