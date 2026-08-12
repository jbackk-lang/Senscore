"""
Testy dla senscoreAll.py -- SENSCORE / TRM / TIMDR / GIA / FIELDCORE.
Uruchomienie: python3 -m pytest test_senscoreAll.py -v
"""
import math
import numpy as np
import pytest

from senscoreAll import (
    Hit, Event, SensorCalibration,
    SENSCOREFilter, TRMFilter, TIMDRFilter, GIAFilter, FIELDCOREFilter,
    FullPipeline,
)


def mkhit(sensor_id=1, x=0.0, y=0.0, z=0.0, t=0.0, energy=1.0, raw_value=1.0):
    return Hit(sensor_id=sensor_id, x=x, y=y, z=z, t=t, energy=energy, raw_value=raw_value)


class TestSENSCORE:
    def test_weight_formula(self):
        cal = {1: SensorCalibration(1, sensitivity=1.0, noise_level=0.1, reliability=0.9)}
        f = SENSCOREFilter(cal)
        h = mkhit(sensor_id=1, energy=10.0, raw_value=2.0)
        out = f.apply(Event([h]))
        expected_weight = 1.0 * 0.9 / (1.0 + 0.1)
        assert out.hits[0].energy == pytest.approx(10.0 * expected_weight)
        assert out.hits[0].raw_value == pytest.approx(2.0 * expected_weight)

    def test_missing_calibration_defaults_to_weight_one(self):
        f = SENSCOREFilter({})
        h = mkhit(sensor_id=99, energy=5.0, raw_value=5.0)
        out = f.apply(Event([h]))
        assert out.hits[0].energy == pytest.approx(5.0)

    def test_empty_event(self):
        f = SENSCOREFilter({})
        out = f.apply(Event([]))
        assert out.hits == []

    def test_zero_reliability_zeroes_signal(self):
        cal = {1: SensorCalibration(1, sensitivity=1.0, noise_level=0.0, reliability=0.0)}
        f = SENSCOREFilter(cal)
        out = f.apply(Event([mkhit(sensor_id=1, energy=100.0)]))
        assert out.hits[0].energy == 0.0


class TestTRM:
    def test_isolated_hit_removed(self):
        f = TRMFilter(max_distance=1.0, max_dt=1.0)
        hits = [mkhit(x=0, t=0), mkhit(x=100, t=100)]
        out = f.apply(Event(hits))
        assert out.hits == []

    def test_clustered_hits_kept(self):
        f = TRMFilter(max_distance=5.0, max_dt=5.0)
        hits = [mkhit(x=0, t=0), mkhit(x=1, t=1), mkhit(x=2, t=2)]
        out = f.apply(Event(hits))
        assert len(out.hits) == 3

    def test_single_hit_always_dropped(self):
        f = TRMFilter()
        out = f.apply(Event([mkhit()]))
        assert out.hits == []

    def test_empty_event(self):
        f = TRMFilter()
        out = f.apply(Event([]))
        assert out.hits == []


class TestTIMDR:
    def test_energy_outlier_removed(self):
        f = TIMDRFilter(max_energy_deviation=2.0, max_time_spread=1000.0)
        hits = [mkhit(energy=10, t=i) for i in range(10)]
        hits.append(mkhit(energy=10000, t=5))
        out = f.apply(Event(hits))
        energies = [h.energy for h in out.hits]
        assert 10000 not in energies

    def test_time_tails_trimmed_when_span_too_large(self):
        f = TIMDRFilter(max_energy_deviation=100.0, max_time_spread=5.0)
        hits = [mkhit(energy=10.0, t=float(i)) for i in range(10)]
        out = f.apply(Event(hits))
        assert len(out.hits) < len(hits)

    def test_time_not_trimmed_when_span_small(self):
        f = TIMDRFilter(max_energy_deviation=100.0, max_time_spread=50.0)
        hits = [mkhit(energy=10.0, t=float(i)) for i in range(10)]
        out = f.apply(Event(hits))
        assert len(out.hits) == len(hits)

    def test_empty_event(self):
        f = TIMDRFilter()
        out = f.apply(Event([]))
        assert out.hits == []

    def test_constant_energy_no_div_by_zero(self):
        f = TIMDRFilter(max_energy_deviation=1.0)
        hits = [mkhit(energy=5.0, t=i) for i in range(5)]
        out = f.apply(Event(hits))
        assert len(out.hits) == 5


class TestGIA:
    def test_fewer_than_three_hits_passthrough(self):
        f = GIAFilter()
        hits = [mkhit(x=0), mkhit(x=1)]
        out = f.apply(Event(hits))
        assert len(out.hits) == 2

    def test_linear_track_kept(self):
        f = GIAFilter(max_residual=0.5)
        hits = [mkhit(x=float(i), y=0.0, z=0.0) for i in range(10)]
        out = f.apply(Event(hits))
        assert len(out.hits) == 10

    def test_off_axis_point_removed(self):
        f = GIAFilter(max_residual=1.0)
        hits = [mkhit(x=float(i), y=0.0, z=0.0) for i in range(10)]
        hits.append(mkhit(x=5.0, y=50.0, z=0.0))
        out = f.apply(Event(hits))
        assert len(out.hits) == 10

    def test_identical_points_no_crash(self):
        f = GIAFilter(max_residual=1.0)
        hits = [mkhit(x=0.0, y=0.0, z=0.0) for _ in range(5)]
        out = f.apply(Event(hits))
        assert len(out.hits) == 5

    def test_residuals_are_real_not_complex(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(3, 15)
            hits = [
                mkhit(x=float(rng.normal(0, 10)), y=float(rng.normal(0, 10)),
                      z=float(rng.normal(0, 10)))
                for _ in range(n)
            ]
            f = GIAFilter(max_residual=3.0)
            out = f.apply(Event(hits))
            assert isinstance(out.hits, list)

    def test_empty_event(self):
        f = GIAFilter()
        out = f.apply(Event([]))
        assert out.hits == []


class TestFIELDCORE:
    def test_default_threshold_rarely_triggers(self):
        f = FIELDCOREFilter()
        hits = [mkhit(energy=10.0) for _ in range(20)]
        hits.append(mkhit(energy=10000.0))
        out = f.apply(Event(hits))
        assert len(out.hits) == len(hits)

    def test_explicit_low_threshold_removes_hot_pixel(self):
        f = FIELDCOREFilter(max_isolated_energy=2.0)
        hits = [mkhit(energy=10.0) for _ in range(20)]
        hits.append(mkhit(energy=10000.0))
        out = f.apply(Event(hits))
        assert len(out.hits) == 20

    def test_empty_event(self):
        f = FIELDCOREFilter()
        out = f.apply(Event([]))
        assert out.hits == []


class TestFullPipeline:
    def test_readme_example_runs(self):
        calibration = {
            1: SensorCalibration(1, sensitivity=1.0, noise_level=0.1, reliability=0.9),
            2: SensorCalibration(2, sensitivity=0.8, noise_level=0.5, reliability=0.7),
        }
        hits = [
            mkhit(sensor_id=1, x=0, y=0, z=0, t=0, energy=10, raw_value=1),
            mkhit(sensor_id=1, x=1, y=0.1, z=0, t=1, energy=11, raw_value=1.1),
            mkhit(sensor_id=2, x=50, y=50, z=0, t=0.5, energy=0.5, raw_value=0.2),
        ]
        event = Event(hits=hits)
        pipeline = FullPipeline(calibration_map=calibration)
        out = pipeline.process(event)
        assert isinstance(out, Event)
        assert len(out.hits) <= len(hits)

    def test_empty_event_through_pipeline(self):
        pipeline = FullPipeline(calibration_map={})
        out = pipeline.process(Event([]))
        assert out.hits == []

    def test_single_hit_event_ends_up_empty(self):
        pipeline = FullPipeline(calibration_map={})
        out = pipeline.process(Event([mkhit(energy=10.0)]))
        assert out.hits == []

    def test_large_random_event_no_crash(self):
        rng = np.random.default_rng(7)
        calibration = {
            i: SensorCalibration(i, sensitivity=rng.uniform(0.5, 1.5),
                                  noise_level=rng.uniform(0, 1),
                                  reliability=rng.uniform(0.5, 1.0))
            for i in range(1, 6)
        }
        hits = [
            mkhit(sensor_id=int(rng.integers(1, 6)),
                  x=float(rng.normal(0, 20)), y=float(rng.normal(0, 20)),
                  z=float(rng.normal(0, 20)), t=float(rng.uniform(0, 30)),
                  energy=float(abs(rng.normal(10, 5))),
                  raw_value=float(abs(rng.normal(1, 0.5))))
            for _ in range(200)
        ]
        pipeline = FullPipeline(calibration_map=calibration)
        out = pipeline.process(Event(hits=hits))
        assert isinstance(out, Event)
        for h in out.hits:
            assert not math.isnan(h.energy)
