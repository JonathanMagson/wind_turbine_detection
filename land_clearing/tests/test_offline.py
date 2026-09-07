"""Offline checks that do not require Earth Engine authentication.

The Earth Engine graph itself cannot be built without credentials (the client
fetches algorithm signatures from the server on first use), so these cover the
pure-Python surface only: AOI presets, stratum bookkeeping, argument parsing,
threshold sanity and the metric maths. Run the notebooks or the ``detect``
entry points against a real project to exercise the rest.

    python -m land_clearing.tests.test_offline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from land_clearing.ccdc import collections as colls
from land_clearing.ccdc import detect as ccdc_detect
from land_clearing.ccdc import rules
from land_clearing.common import aois, masks, validation
from land_clearing.omnibus_s1 import detect as omnibus_detect


# --- shared AOI / stratum bookkeeping ---

def test_aoi_presets_are_valid_nsw_boxes():
    assert aois.AOIS
    for name, (w, s, e, n) in aois.AOIS.items():
        assert w < e and s < n, name
        # NSW spans roughly 141-154E, 28-38S.
        assert 140.0 < w < 154.0 and 140.0 < e < 154.0, name
        assert -38.0 < s < -28.0 and -38.0 < n < -28.0, name


def test_aoi_geometry_rejects_unknown_name():
    try:
        aois.aoi_geometry(name='sydney_opera_house')
    except ValueError as exc:
        assert 'Unknown AOI' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_stratum_codes_are_distinct_and_named():
    codes = [masks.FOREST, masks.WOODLAND, masks.GRASSLAND, masks.CROPLAND]
    assert len(set(codes)) == len(codes)
    assert 0 not in codes, 'zero is reserved for "outside all strata"'
    for c in codes:
        assert c in masks.STRATUM_NAMES


def test_woody_mask_rejects_unknown_kind():
    try:
        masks.woody_mask('lidar')
    except ValueError as exc:
        assert 'Unknown woody mask' in str(exc)
    else:
        raise AssertionError('expected ValueError')


# --- CCDC pipeline ---

def test_ccdc_band_lists_match_scaling_docs():
    assert colls.S1_BANDS == ['VV', 'VH', 'RATIO']
    assert colls.S2_BANDS == ['NDVI', 'NBR', 'BSI']
    assert colls.S1_SCALE == 100
    assert colls.S2_SCALE == 10000


def test_thresholds_have_the_signs_the_rules_assume():
    # Vegetation loss lowers VH and NDVI, and raises exposed soil.
    assert rules.VH_DROP_WOODY < 0
    assert rules.VH_DROP_GRASS < 0
    assert rules.NDVI_DROP < 0
    assert rules.BSI_RISE > 0
    # A burn drops NBR further than clearing does, so the fire screen sits
    # below the vegetation-loss threshold.
    assert rules.NBR_FIRE_DROP < rules.NDVI_DROP
    assert 0.0 < rules.CHANGE_PROB <= 1.0


def test_clearing_rules_requires_at_least_one_sensor():
    try:
        rules.clearing_rules(s1_breaks=None, s2_breaks=None)
    except ValueError as exc:
        assert 'at least one' in str(exc).lower()
    else:
        raise AssertionError('expected ValueError')


def test_ccdc_detect_defaults():
    args = ccdc_detect.parse_args(
        ['--aoi', 'moree', '--start', '2023-01-01', '--end', '2024-01-01'])
    assert args.sensors == 'both'
    assert args.history_years == 2.0
    assert args.min_observations == 4
    assert args.include_cropland is False
    assert args.no_fire_screen is False
    assert args.min_mmu_ha == 0.5


def test_ccdc_detect_lambda_flag_maps_to_lambda_underscore():
    # "lambda" is a Python keyword, so the flag has to land on a legal name.
    args = ccdc_detect.parse_args(
        ['--aoi', 'moree', '--start', '2023-01-01', '--end', '2024-01-01',
         '--lambda-', '0.005'])
    assert args.lambda_ == 0.005


def test_ccdc_detect_accepts_sensor_subsets():
    for sensors in ('both', 's1', 's2'):
        args = ccdc_detect.parse_args(
            ['--bbox', '149.6', '-29.6', '150.0', '-29.3',
             '--start', '2023-01-01', '--end', '2024-01-01',
             '--sensors', sensors])
        assert args.sensors == sensors
        assert args.aoi is None


# --- omnibus pipeline ---

def test_omnibus_detect_defaults():
    args = omnibus_detect.parse_args(
        ['--aoi', 'moree', '--start', '2023-01-01', '--end', '2024-01-01'])
    assert args.orbit_pass == 'DESCENDING'
    assert args.woody_mask == 'worldcover'
    assert args.alpha == 0.01
    assert args.min_mmu_ha == 0.5
    assert args.export == 'none'


def test_omnibus_detect_accepts_bbox():
    args = omnibus_detect.parse_args(
        ['--bbox', '149.6', '-29.6', '150.0', '-29.3',
         '--start', '2023-01-01', '--end', '2024-01-01'])
    assert args.aoi is None
    assert args.bbox == [149.6, -29.6, 150.0, -29.3]


# --- validation maths ---

def test_validation_argument_parsing():
    args = validation.parse_args(
        ['--detected', 'projects/p/assets/x', '--aoi', 'moree', '--year', '2023'])
    assert args.reference == 'hansen'
    assert args.year == 2023
    assert args.scale == 30


def test_metrics_perfect_agreement():
    m = validation.metrics({'tp': 100.0, 'fp': 0.0, 'fn': 0.0, 'tn': 900.0})
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['f1'] == 1.0
    assert m['kappa'] == 1.0


def test_metrics_known_case():
    m = validation.metrics({'tp': 50.0, 'fp': 10.0, 'fn': 20.0, 'tn': 920.0})
    assert abs(m['precision'] - 50 / 60) < 1e-4
    assert abs(m['recall'] - 50 / 70) < 1e-4
    assert abs(m['accuracy'] - 0.97) < 1e-4
    assert 0.0 < m['kappa'] < 1.0


def test_metrics_no_detections():
    m = validation.metrics({'tp': 0.0, 'fp': 0.0, 'fn': 10.0, 'tn': 990.0})
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['f1'] == 0.0


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS %s' % name)
            except Exception as exc:                     # noqa: BLE001
                failures += 1
                print('FAIL %s: %s' % (name, exc))
    print('%d failure(s)' % failures)
    raise SystemExit(1 if failures else 0)
