#!/usr/bin/env python3
"""End-to-end Sentinel-1 land-clearing detection for a NSW AOI.

Woody vegetation only -- see ``land_clearing.ccdc`` for a product that also
covers grassland.

Run from the repository root:
    python -m land_clearing.omnibus_s1.detect --aoi moree --start 2023-01-01 --end 2024-01-01 \
        --project my-gee-project --export drive

Everything except the small summary reductions runs server-side in Earth Engine,
so no imagery is downloaded.
"""

import argparse
import json
import sys

import ee

from ..common import aois, masks, stats
from . import clearing as clearing_mod
from . import omnibus
from . import s1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    aoi = p.add_mutually_exclusive_group(required=True)
    aoi.add_argument('--aoi', choices=sorted(aois.AOIS),
                     help='named NSW AOI preset')
    aoi.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'),
                     help='explicit bounding box in EPSG:4326')
    p.add_argument('--start', required=True, help='start date, YYYY-MM-DD')
    p.add_argument('--end', required=True, help='end date, YYYY-MM-DD (exclusive)')
    p.add_argument('--project', help='Google Cloud project for ee.Initialize')
    p.add_argument('--orbit-pass', default='DESCENDING',
                   choices=['ASCENDING', 'DESCENDING'])
    p.add_argument('--relative-orbit', type=int,
                   help='force a relative orbit instead of the most frequent one')
    p.add_argument('--stride', type=int, default=1,
                   help='keep every Nth acquisition (default 1)')
    p.add_argument('--alpha', type=float, default=0.01,
                   help='per-test significance level (default 0.01)')
    p.add_argument('--median', action='store_true',
                   help='5x5 median filter the change maps')
    p.add_argument('--woody-mask', default='worldcover', choices=masks.WOODY_MASKS)
    p.add_argument('--tree-cover-pct', type=int, default=20,
                   help='canopy threshold for the hansen mask (default 20)')
    p.add_argument('--min-mmu-ha', type=float, default=0.5,
                   help='minimum mapping unit in hectares (default 0.5, 0 to disable)')
    p.add_argument('--scale', type=int, default=10, help='output scale in metres')
    p.add_argument('--export', choices=['none', 'drive', 'asset'], default='none')
    p.add_argument('--export-prefix', default='nsw_clearing',
                   help='file/asset name prefix for exports')
    p.add_argument('--asset-folder',
                   help='asset path prefix, e.g. projects/my-project/assets')
    p.add_argument('--out', default='clearing_summary.json',
                   help='where to write the run summary (default clearing_summary.json)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    aoi = aois.aoi_geometry(name=args.aoi, bbox=args.bbox)

    im_list, dates, rel_orbit = s1.build_series(
        aoi, args.start, args.end,
        orbit_pass=args.orbit_pass,
        relative_orbit=args.relative_orbit,
        stride=args.stride)
    n_images = len(dates)
    n_intervals = n_images - 1
    print('%d acquisitions on relative orbit %d (%s), %s to %s'
          % (n_images, rel_orbit, args.orbit_pass, dates[0], dates[-1]),
          file=sys.stderr)

    result = ee.Dictionary(
        omnibus.change_maps(im_list, median=args.median, alpha=args.alpha))
    bmap = ee.Image(result.get('bmap'))

    mask = masks.woody_mask(args.woody_mask, tree_cover_pct=args.tree_cover_pct)
    clearing = clearing_mod.clearing_from_bmap(
        bmap, n_intervals, mask=mask, min_mmu_ha=args.min_mmu_ha,
        scale=args.scale)

    total_ha = stats.cleared_area_ha(clearing, aoi, scale=args.scale)
    per_interval = stats.area_by_interval(clearing, aoi, n_intervals,
                                        scale=args.scale)

    intervals = [{'interval': i,
                  'from': dates[i - 1],
                  'to': dates[i],
                  'cleared_ha': round(per_interval.get(i, 0.0), 3)}
                 for i in range(1, n_intervals + 1)]

    summary = {
        'aoi': args.aoi or list(args.bbox),
        'bbox': aoi.bounds().getInfo()['coordinates'],
        'start': args.start,
        'end': args.end,
        'orbit_pass': args.orbit_pass,
        'relative_orbit': rel_orbit,
        'n_acquisitions': n_images,
        'stride': args.stride,
        'alpha': args.alpha,
        'median_filter': args.median,
        'woody_mask': args.woody_mask,
        'min_mmu_ha': args.min_mmu_ha,
        'scale_m': args.scale,
        'total_cleared_ha': round(total_ha, 3),
        'dates': dates,
        'intervals': intervals,
    }

    with open(args.out, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ('relative_orbit', 'n_acquisitions', 'total_cleared_ha')},
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
