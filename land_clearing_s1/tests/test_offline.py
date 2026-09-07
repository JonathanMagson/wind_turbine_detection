"""Offline checks that do not require Earth Engine authentication.

The Earth Engine graph itself cannot be built without credentials (the client
fetches algorithm signatures from the server on first use), so these cover the
pure-Python surface only: AOI presets, argument parsing and the metric maths.
Run the notebook or ``detect.py`` against a real project to exercise the rest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect
import nsw
import validate


def test_aoi_presets_are_valid_nsw_boxes():
    assert nsw.AOIS
    for name, (w, s, e, n) in nsw.AOIS.items():
        assert w < e and s < n, name
        # NSW spans roughly 141-154E, 28-38S.
        assert 140.0 < w < 154.0 and 140.0 < e < 154.0, name
        assert -38.0 < s < -28.0 and -38.0 < n < -28.0, name


def test_aoi_geometry_rejects_unknown_name():
    try:
        nsw.aoi_geometry(name='sydney_opera_house')
    except ValueError as exc:
        assert 'Unknown AOI' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_woody_mask_rejects_unknown_kind():
    try:
        nsw.woody_mask('lidar')
    except ValueError as exc:
        assert 'Unknown woody mask' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_detect_argument_defaults():
    args = detect.parse_args(
        ['--aoi', 'moree', '--start', '2023-01-01', '--end', '2024-01-01'])
    assert args.aoi == 'moree'
    assert args.orbit_pass == 'DESCENDING'
    assert args.woody_mask == 'worldcover'
    assert args.alpha == 0.01
    assert args.min_mmu_ha == 0.5
    assert args.export == 'none'


def test_detect_accepts_bbox_instead_of_preset():
    args = detect.parse_args(
        ['--bbox', '149.6', '-29.6', '150.0', '-29.3',
         '--start', '2023-01-01', '--end', '2024-01-01'])
    assert args.aoi is None
    assert args.bbox == [149.6, -29.6, 150.0, -29.3]


def test_validate_argument_parsing():
    args = validate.parse_args(
        ['--detected', 'projects/p/assets/x', '--aoi', 'moree', '--year', '2023'])
    assert args.reference == 'hansen'
    assert args.year == 2023
    assert args.scale == 30


def test_metrics_perfect_agreement():
    m = validate.metrics({'tp': 100.0, 'fp': 0.0, 'fn': 0.0, 'tn': 900.0})
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['f1'] == 1.0
    assert m['kappa'] == 1.0


def test_metrics_known_case():
    m = validate.metrics({'tp': 50.0, 'fp': 10.0, 'fn': 20.0, 'tn': 920.0})
    assert abs(m['precision'] - 50 / 60) < 1e-4
    assert abs(m['recall'] - 50 / 70) < 1e-4
    assert abs(m['accuracy'] - 0.97) < 1e-4
    assert 0.0 < m['kappa'] < 1.0


def test_metrics_no_detections():
    m = validate.metrics({'tp': 0.0, 'fp': 0.0, 'fn': 10.0, 'tn': 990.0})
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
    raise SystemExit(1 if failures else 0)
