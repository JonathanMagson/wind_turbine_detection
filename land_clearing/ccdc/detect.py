#!/usr/bin/env python3
"""End-to-end CCDC land-clearing detection across forest, woodland and grassland.

Run from the repository root:

    python -m land_clearing.ccdc.detect --aoi moree \
        --start 2023-01-01 --end 2024-01-01 --project my-gee-project

CCDC needs history before the reporting window to fit each pixel's seasonal
model, so the collections are built from ``--history-years`` before ``--start``.
Breaks are then only reported inside ``[--start, --end)``.
"""

import argparse
import json
import sys

import ee

from ..common import aois, masks, stats
from . import collections as colls
from . import rules as rules_mod
from . import segmentation


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    aoi = p.add_mutually_exclusive_group(required=True)
    aoi.add_argument('--aoi', choices=sorted(aois.AOIS))
    aoi.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'))
    p.add_argument('--start', required=True, help='reporting window start, YYYY-MM-DD')
    p.add_argument('--end', required=True, help='reporting window end, YYYY-MM-DD')
    p.add_argument('--history-years', type=float, default=2.0,
                   help='years of history before --start used to fit the model '
                        '(default 2.0; below ~1.5 the harmonics are unreliable)')
    p.add_argument('--project', help='Google Cloud project for ee.Initialize')
    p.add_argument('--sensors', default='both', choices=['both', 's1', 's2'],
                   help='which stacks to segment (default both). Grassland '
                        'cannot be assessed with s1 alone.')
    p.add_argument('--orbit-pass', default='DESCENDING',
                   choices=['ASCENDING', 'DESCENDING'])
    p.add_argument('--relative-orbit', type=int,
                   help='force a relative orbit; 0 disables the restriction')
    p.add_argument('--multilook-px', type=int, default=3,
                   help='boxcar multi-look window for S1 in pixels (default 3)')
    p.add_argument('--max-cloud-pct', type=int, default=60,
                   help='scene-level S2 cloud cover cutoff (default 60)')
    p.add_argument('--min-observations', type=int, default=4,
                   help='consecutive off-model observations to confirm a break')
    p.add_argument('--chi-square-probability', type=float, default=0.99)
    p.add_argument('--lambda-', type=float, default=20.0 / 10000.0,
                   dest='lambda_', help='CCDC regularisation (default 0.002)')
    p.add_argument('--change-prob', type=float, default=rules_mod.CHANGE_PROB,
                   help='minimum CCDC change probability (default 0.90)')
    p.add_argument('--vh-drop-woody', type=float, default=rules_mod.VH_DROP_WOODY,
                   help='VH magnitude threshold in dB x 100 (default -150)')
    p.add_argument('--ndvi-drop', type=float, default=rules_mod.NDVI_DROP,
                   help='NDVI magnitude threshold x 10000 (default -1500)')
    p.add_argument('--bsi-rise', type=float, default=rules_mod.BSI_RISE,
                   help='BSI magnitude threshold x 10000 (default 800)')
    p.add_argument('--no-fire-screen', action='store_true',
                   help='keep detections whose NBR drop suggests a burn')
    p.add_argument('--include-cropland', action='store_true',
                   help='assess cropland too (off by default: crop cycles are '
                        'not clearing)')
    p.add_argument('--min-mmu-ha', type=float, default=0.5,
                   help='minimum mapping unit in hectares (0 disables)')
    p.add_argument('--scale', type=int, default=10)
    p.add_argument('--export', choices=['none', 'drive', 'asset'], default='none')
    p.add_argument('--export-prefix', default='nsw_clearing_ccdc')
    p.add_argument('--asset-folder')
    p.add_argument('--out', default='ccdc_summary.json')
    return p.parse_args(argv)


def _shift_years(date_str, years):
    return ee.Date(date_str).advance(-years, 'year').format('YYYY-MM-dd').getInfo()


def main(argv=None):
    args = parse_args(argv)

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    aoi = aois.aoi_geometry(name=args.aoi, bbox=args.bbox)
    fit_start = _shift_years(args.start, args.history_years)
    start_year = segmentation.year_fraction(args.start)
    end_year = segmentation.year_fraction(args.end)

    print('fitting from %s, reporting %s to %s'
          % (fit_start, args.start, args.end), file=sys.stderr)

    s1_breaks = s2_breaks = None
    rel_orbit = None
    n_s1 = n_s2 = 0

    if args.sensors in ('both', 's1'):
        s1_coll, rel_orbit = colls.s1_stack(
            aoi, fit_start, args.end, orbit_pass=args.orbit_pass,
            relative_orbit=args.relative_orbit,
            multilook_px=args.multilook_px)
        n_s1 = s1_coll.size().getInfo()
        print('S1: %d images on relative orbit %s' % (n_s1, rel_orbit),
              file=sys.stderr)
        s1_ccdc = segmentation.run_ccdc(
            s1_coll, colls.S1_BANDS,
            min_observations=args.min_observations,
            chi_square_probability=args.chi_square_probability,
            lambda_=args.lambda_)
        s1_breaks = segmentation.extract_breaks(
            s1_ccdc, colls.S1_BANDS, start_year, end_year, prefix='s1_')

    if args.sensors in ('both', 's2'):
        s2_coll = colls.s2_stack(aoi, fit_start, args.end,
                                 max_cloud_pct=args.max_cloud_pct)
        n_s2 = s2_coll.size().getInfo()
        print('S2: %d images' % n_s2, file=sys.stderr)
        s2_ccdc = segmentation.run_ccdc(
            s2_coll, colls.S2_BANDS,
            min_observations=args.min_observations,
            chi_square_probability=args.chi_square_probability,
            lambda_=args.lambda_)
        s2_breaks = segmentation.extract_breaks(
            s2_ccdc, colls.S2_BANDS, start_year, end_year, prefix='s2_')

    if args.sensors == 's1':
        print('WARNING: Sentinel-1 only. Grassland has no volume scattering to '
              'lose, so its conversion cannot be established from radar alone '
              'and is reported as zero, not as absence of clearing.',
              file=sys.stderr)

    strata = masks.cover_strata(include_cropland=args.include_cropland)
    clearing = rules_mod.clearing_rules(
        s1_breaks=s1_breaks, s2_breaks=s2_breaks, strata=strata,
        vh_drop_woody=args.vh_drop_woody, ndvi_drop=args.ndvi_drop,
        bsi_rise=args.bsi_rise, change_prob=args.change_prob,
        screen_fire=not args.no_fire_screen)
    clearing = rules_mod.apply_mmu(clearing, min_mmu_ha=args.min_mmu_ha,
                                  scale=args.scale)

    total_ha = stats.cleared_area_ha(clearing, aoi, scale=args.scale)
    by_stratum = stats.area_by_stratum(clearing, aoi, scale=args.scale)
    y0, y1 = int(args.start[:4]), int(args.end[:4])
    by_year = stats.area_by_year(clearing, aoi, y0, y1, scale=args.scale)

    summary = {
        'method': 'CCDC (Zhu & Woodcock 2014) via ee.Algorithms.TemporalSegmentation.Ccdc',
        'aoi': args.aoi or list(args.bbox),
        'fit_start': fit_start,
        'start': args.start,
        'end': args.end,
        'sensors': args.sensors,
        'n_s1_images': n_s1,
        'n_s2_images': n_s2,
        'relative_orbit': rel_orbit,
        'min_observations': args.min_observations,
        'chi_square_probability': args.chi_square_probability,
        'change_prob': args.change_prob,
        'thresholds': {'vh_drop_woody': args.vh_drop_woody,
                       'ndvi_drop': args.ndvi_drop,
                       'bsi_rise': args.bsi_rise},
        'fire_screen': not args.no_fire_screen,
        'include_cropland': args.include_cropland,
        'min_mmu_ha': args.min_mmu_ha,
        'scale_m': args.scale,
        'total_cleared_ha': round(total_ha, 3),
        'cleared_ha_by_stratum': {k: round(v, 3) for k, v in by_stratum.items()},
        'cleared_ha_by_year': {str(k): round(v, 3) for k, v in by_year.items()},
    }
    if args.sensors == 's1':
        summary['caveat'] = ('Sentinel-1 only: grassland conversion is not '
                             'assessable from radar alone and is reported as 0.')

    with open(args.out, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({'total_cleared_ha': summary['total_cleared_ha'],
                      'cleared_ha_by_stratum': summary['cleared_ha_by_stratum']},
                     indent=2))
    print('Wrote %s' % args.out, file=sys.stderr)

    if args.export != 'none':
        name = '%s_%s_%s_%s' % (args.export_prefix, args.aoi or 'bbox',
                                args.start, args.end)
        common = dict(image=clearing.clip(aoi), description=name,
                      region=aoi, scale=args.scale, maxPixels=int(1e13))
        if args.export == 'drive':
            task = ee.batch.Export.image.toDrive(fileNamePrefix=name, **common)
        else:
            if not args.asset_folder:
                raise SystemExit('--export asset requires --asset-folder')
            task = ee.batch.Export.image.toAsset(
                assetId='%s/%s' % (args.asset_folder.rstrip('/'), name), **common)
        task.start()
        print('Started export task %s (%s)' % (task.id, name), file=sys.stderr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
