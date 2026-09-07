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
from land_clearing.local_s1 import catalog, omnibus_np, worldcover
from land_clearing.local_s1 import run as local_run
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


# --- local (no-Earth-Engine) pipeline ---

def test_relative_orbit_matches_known_scene():
    # S1A_IW_GRDH_1SDV_20230106T192310_..._046667_... covers the NSW AOI on
    # track 45; absolute orbit 46667 must map back to it.
    assert catalog.relative_orbit(46667, 'S1A') == 45


def test_scene_time_parses_utc_from_name():
    p = ('GRD/2023/1/6/IW/DV/S1A_IW_GRDH_1SDV_20230106T192310_'
         '20230106T192335_046667_0597F6_EBEB/')
    assert catalog.scene_time(p) == 192310


def test_worldcover_tile_naming():
    # NSW AOI at 148.8E, 30.0S sits in the 3-degree tile with SW corner S30E147.
    assert worldcover.tile_name(148.80, -29.95) == \
        'ESA_WorldCover_10m_2021_v200_S30E147_Map.tif'
    assert worldcover.tile_name(0.5, 0.5) == \
        'ESA_WorldCover_10m_2021_v200_N00E000_Map.tif'


def test_vectorise_preserves_area_of_diagonally_touching_pixels():
    # label() uses 8-connectivity, so rasterio's shapes() must too. With the
    # default connectivity=4, a component whose parts touch only at a corner is
    # split into separate polygons and keeping the largest silently drops area.
    import numpy as np
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from scipy.ndimage import label
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = True      # a diagonal chain
    lab, n = label(mask, structure=np.ones((3, 3)))
    assert n == 1, 'expected one 8-connected component'

    transform = from_origin(0, 0, 1, 1)
    parts = [shapely_shape(g) for g, v in
             shapes(lab.astype(np.int32), mask=mask, transform=transform,
                    connectivity=8) if int(v) == 1]
    assert abs(unary_union(parts).area - 3.0) < 1e-9, 'area must equal 3 pixels'


def test_worldcover_tiles_for_bbox_spans_boundaries():
    # A 3-degree tile boundary runs along latitude -30, straight through the
    # Brigalow Belt, so a NSW AOI can straddle two tiles. Reading only the
    # centre tile silently returns wrong land cover for the other half.
    straddling = (149.50, -30.05, 149.70, -29.85)
    tiles = worldcover.tiles_for_bbox(straddling)
    assert len(tiles) == 2, tiles
    names = {worldcover._tile_name_from_corner(*t) for t in tiles}
    assert names == {'ESA_WorldCover_10m_2021_v200_S30E147_Map.tif',
                     'ESA_WorldCover_10m_2021_v200_S33E147_Map.tif'}


def test_worldcover_tiles_for_bbox_single_tile():
    inside = (149.50, -30.55, 149.70, -30.35)
    assert len(worldcover.tiles_for_bbox(inside)) == 1


def test_worldcover_tiles_for_bbox_spans_four():
    # Crossing both a longitude and a latitude boundary needs all four.
    assert len(worldcover.tiles_for_bbox((148.10, -31.15, 150.95, -29.20))) == 4


def test_omnibus_np_detects_an_injected_change_at_the_right_interval():
    import numpy as np
    rng = np.random.default_rng(0)
    H = W = 24
    k = 10
    enl = omnibus_np.ENL
    change_at = 5          # images 0..4 stable, 5..9 cleared

    def speckle(vv, vh, shape):
        return np.stack([rng.gamma(enl, vv / enl, shape) * enl,
                         rng.gamma(enl, vh / enl, shape) * enl])

    series = []
    for t in range(k):
        im = speckle(0.10, 0.030, (H, W))
        if t >= change_at:
            im[:, :, W // 2:] = speckle(0.055, 0.008, (H, W // 2))
        series.append(im)

    res = omnibus_np.change_maps(series, alpha=0.01)
    neg = res['bmap'] == omnibus_np.NEGATIVE

    cleared, stable = neg[:, :, W // 2:], neg[:, :, :W // 2]
    # The injected change must dominate its own interval.
    per_interval = cleared.sum(axis=(1, 2))
    assert per_interval.argmax() == change_at - 1, per_interval
    assert cleared.any(0).mean() > 0.3, 'detection rate too low'
    # False positives must stay near the significance level.
    assert stable.any(0).mean() < 0.06, 'false positive rate too high'


def test_first_negative_interval_takes_the_earliest():
    import numpy as np
    bmap = np.zeros((3, 2, 2))
    bmap[2, 0, 0] = omnibus_np.NEGATIVE
    bmap[0, 0, 0] = omnibus_np.NEGATIVE      # earlier one must win
    bmap[1, 1, 1] = omnibus_np.POSITIVE      # wrong direction, ignored
    out = local_run.first_negative_interval(bmap)
    assert out[0, 0] == 1
    assert out[1, 1] == 0


def test_apply_mmu_drops_small_components():
    import numpy as np
    mask = np.zeros((20, 20), dtype=bool)
    mask[2, 2] = True                 # 1 px speck
    mask[10:14, 10:14] = True         # 16 px block
    pixel_area_ha = 0.01              # 100 m^2 per pixel
    kept = local_run.apply_mmu(mask, min_mmu_ha=0.05, pixel_area_ha=pixel_area_ha)
    assert not kept[2, 2]
    assert kept[11, 11]
    assert kept.sum() == 16


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
