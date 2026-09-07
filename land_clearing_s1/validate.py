#!/usr/bin/env python3
"""Validate detected clearing against reference data.

Two reference options:

``--reference hansen``
    Hansen Global Forest Change ``lossyear``, available directly in the GEE
    catalogue. Convenient and fully automatic, but it is a poor yardstick in
    NSW: the product is tuned to closed-canopy forest and systematically misses
    the sparse woody vegetation that most NSW clearing removes. Treat agreement
    as a sanity check, not an accuracy figure.

``--reference asset``
    A ``FeatureCollection`` you upload yourself. The right reference for NSW is
    the SLATS (Statewide Landcover and Trees Study) woody vegetation change
    layer, published on the NSW SEED portal:
    https://datasets.seed.nsw.gov.au/  (search "SLATS woody vegetation change").
    Download the year(s) covering your run, clip to the AOI, upload as a GEE
    table asset, then pass its asset id here.

Reported metrics are pixel-count based within the AOI: true/false positives and
negatives, precision, recall, F1 and Cohen's kappa. Because the reference is
annual and the detection is per-interval, compare a detection run whose date
range matches the reference year.
"""

import argparse
import json
import sys

import ee

import nsw


def hansen_reference(year, aoi):
    """Hansen GFC loss for a calendar year as a binary image."""
    gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    return gfc.select('lossyear').eq(year - 2000).clip(aoi).rename('reference')


def asset_reference(asset_id, aoi):
    """Rasterise a reference FeatureCollection to a binary image."""
    fc = ee.FeatureCollection(asset_id).filterBounds(aoi)
    return (fc.map(lambda f: f.set('ref', 1))
            .reduceToImage(['ref'], ee.Reducer.first())
            .unmask(0).gt(0).clip(aoi).rename('reference'))


def confusion(detected, reference, aoi, scale=30, max_pixels=1e13):
    """Pixel confusion counts between two binary images."""
    d = ee.Image(detected).select('clearing').unmask(0).gt(0)
    r = ee.Image(reference).unmask(0).gt(0)
    counts = ee.Image.cat(
        d.And(r).rename('tp'),
        d.And(r.Not()).rename('fp'),
        d.Not().And(r).rename('fn'),
        d.Not().And(r.Not()).rename('tn'),
    ).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
                   maxPixels=max_pixels).getInfo()
    return {k: float(counts.get(k, 0) or 0) for k in ('tp', 'fp', 'fn', 'tn')}


def metrics(c):
    tp, fp, fn, tn = c['tp'], c['fp'], c['fn'], c['tn']
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    po = (tp + tn) / n if n else 0.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4), 'accuracy': round(po, 4), 'kappa': round(kappa, 4)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--detected', required=True,
                   help='GEE image asset id of a clearing map exported by detect.py')
    p.add_argument('--reference', choices=['hansen', 'asset'], default='hansen')
    p.add_argument('--reference-asset', help='table asset id when --reference asset')
    p.add_argument('--year', type=int, help='calendar year when --reference hansen')
    aoi = p.add_mutually_exclusive_group(required=True)
    aoi.add_argument('--aoi', choices=sorted(nsw.AOIS))
    aoi.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'))
    p.add_argument('--project')
    p.add_argument('--scale', type=int, default=30,
                   help='comparison scale in metres (default 30, the Hansen grid)')
    p.add_argument('--out', default='validation.json')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    aoi = nsw.aoi_geometry(name=args.aoi, bbox=args.bbox)
    detected = ee.Image(args.detected)

    if args.reference == 'hansen':
        if not args.year:
            raise SystemExit('--reference hansen requires --year')
        reference = hansen_reference(args.year, aoi)
    else:
        if not args.reference_asset:
            raise SystemExit('--reference asset requires --reference-asset')
        reference = asset_reference(args.reference_asset, aoi)

    counts = confusion(detected, reference, aoi, scale=args.scale)
    result = {'detected': args.detected,
              'reference': args.reference_asset or 'hansen %s' % args.year,
              'scale_m': args.scale,
              'counts': counts,
              'metrics': metrics(counts)}

    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print('Wrote %s' % args.out, file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
