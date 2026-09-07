#!/usr/bin/env python3
"""End-to-end Sentinel-1 land-clearing detection with no Earth Engine account.

Reads Sentinel-1 GRD directly from the public AWS mirror, calibrates and
geocodes a small AOI window from each scene, and runs the sequential omnibus
test locally. Everything streams over HTTPS range reads; no scene is downloaded
in full and nothing but the outputs is written to disk.

    python -m land_clearing.local_s1.run --bbox 149.55 -30.05 149.75 -29.85 \
        --start 2023-01-01 --end 2024-01-01 --outdir out/

The P-value array holds O(k^2) full-size float arrays, so the omnibus step runs
in tiles to keep memory bounded regardless of AOI size or series length.
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import catalog, omnibus_np, reader, worldcover


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bbox', type=float, nargs=4, required=True,
                   metavar=('W', 'S', 'E', 'N'))
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--orbit', type=int, help='force a relative orbit')
    p.add_argument('--pixel-deg', type=float, default=0.0001,
                   help='output pixel size in degrees (default 0.0001, ~11 m)')
    p.add_argument('--alpha', type=float, default=0.01,
                   help='per-test significance level (default 0.01)')
    p.add_argument('--median', action='store_true',
                   help='5x5 median filter on the omnibus P values')
    p.add_argument('--multilook', type=int, default=5,
                   help='block-average NxN before testing (default 5, ~56 m). '
                        'At 10 m the per-pixel speckle standard deviation is '
                        '~2.1 dB, which swamps the 1-2 dB step that clearing '
                        'sparse woody vegetation produces; 1 disables.')
    p.add_argument('--enl', type=float,
                   help='override the equivalent number of looks. By default '
                        'it is estimated from the imagery and calibrated '
                        'against the nominal 4.4 of the raw GRD product.')
    p.add_argument('--min-mmu-ha', type=float, default=0.5,
                   help='minimum mapping unit in hectares (0 disables)')
    p.add_argument('--woody-mask', action='store_true', default=True,
                   help='restrict detections to WorldCover tree/shrub (default on)')
    p.add_argument('--no-woody-mask', dest='woody_mask', action='store_false')
    p.add_argument('--tile', type=int, default=384,
                   help='tile size in pixels for the omnibus step')
    p.add_argument('--workers', type=int, default=5,
                   help='parallel scene reads (default 5)')
    p.add_argument('--max-scenes', type=int, help='cap the series length')
    p.add_argument('--outdir', default='out')
    return p.parse_args(argv)


def read_series(scenes, bbox, pixel_deg, workers):
    """Read every scene onto the AOI grid. Returns (stack, kept_scenes, diags)."""
    def one(sc):
        try:
            arr, diag = reader.read_scene(sc, bbox, pixel_deg)
            return sc, arr, diag
        except Exception as exc:                             # noqa: BLE001
            print('  skip %s: %s' % (sc['id'][:44], exc), file=sys.stderr)
            return sc, None, None

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, scenes))

    kept, arrays, diags = [], [], []
    for sc, arr, diag in results:
        if arr is None:
            continue
        kept.append(sc)
        arrays.append(arr.astype(np.float32))
        diags.append(diag)
    return arrays, kept, diags


def tiled_change_maps(stack, alpha, median, enl, tile):
    """Run the omnibus test tile by tile to bound peak memory."""
    k = len(stack)
    h, w = stack[0].shape[1:]
    bmap = np.zeros((k - 1, h, w), dtype=np.uint8)
    smap = np.zeros((h, w), dtype=np.int16)
    fmap = np.zeros((h, w), dtype=np.int16)

    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            r1, c1 = min(r0 + tile, h), min(c0 + tile, w)
            sub = [im[:, r0:r1, c0:c1].astype(np.float64) * enl for im in stack]
            # Speckle-free (masked) pixels would break the log; fill them with
            # the tile median so they simply never test significant.
            for im in sub:
                bad = ~np.isfinite(im) | (im <= 0)
                if bad.any():
                    im[bad] = np.nanmedian(im[~bad]) if (~bad).any() else 1.0
            res = omnibus_np.change_maps(sub, median=median, alpha=alpha, m=enl)
            bmap[:, r0:r1, c0:c1] = res['bmap'].astype(np.uint8)
            smap[r0:r1, c0:c1] = res['smap'].astype(np.int16)
            fmap[r0:r1, c0:c1] = res['fmap'].astype(np.int16)
    return {'bmap': bmap, 'smap': smap, 'fmap': fmap}


def first_negative_interval(bmap):
    """1-based index of the first negative-definite change, 0 where none."""
    neg = (bmap == omnibus_np.NEGATIVE)
    out = np.zeros(bmap.shape[1:], dtype=np.int16)
    for i in range(bmap.shape[0] - 1, -1, -1):
        out[neg[i]] = i + 1
    return out


def apply_mmu(mask, min_mmu_ha, pixel_area_ha):
    """Drop connected components smaller than the minimum mapping unit."""
    if not min_mmu_ha:
        return mask
    from scipy.ndimage import label, sum as ndsum
    min_px = max(1, int(round(min_mmu_ha / pixel_area_ha)))
    if min_px <= 1:
        return mask
    lab, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndsum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def write_geotiff(path, arrays, names, bbox, pixel_deg):
    reader._prepare_gdal_env()
    import rasterio
    from rasterio.transform import from_origin
    w, s, e, n = bbox
    h, wid = arrays[0].shape
    transform = from_origin(w, n, pixel_deg, pixel_deg)
    with rasterio.open(path, 'w', driver='GTiff', height=h, width=wid,
                       count=len(arrays), dtype='int16', crs='EPSG:4326',
                       transform=transform, compress='deflate') as ds:
        for i, (a, nm) in enumerate(zip(arrays, names), start=1):
            ds.write(a.astype(np.int16), i)
            ds.set_band_description(i, nm)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    bbox = tuple(args.bbox)
    from shapely.geometry import box
    aoi = box(*bbox)

    print('searching the AWS Sentinel-1 archive...', file=sys.stderr)
    scenes = catalog.find_series(aoi, args.start, args.end, orbit=args.orbit)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    if len(scenes) < 3:
        raise SystemExit('Only %d scenes found; need at least 3.' % len(scenes))
    track = scenes[0]['relativeOrbit']
    print('%d scenes on relative orbit %d' % (len(scenes), track), file=sys.stderr)

    print('reading and geocoding...', file=sys.stderr)
    stack, scenes, diags = read_series(scenes, bbox, args.pixel_deg, args.workers)
    if len(stack) < 3:
        raise SystemExit('Only %d scenes read successfully.' % len(stack))
    dates = [s['startTime'][:10] for s in scenes]
    print('stack: %d dates, %d x %d px at %.5f deg'
          % (len(stack), stack[0].shape[1], stack[0].shape[2], args.pixel_deg),
          file=sys.stderr)

    # Multi-look, and work out how many looks that actually bought. The naive
    # answer (nominal ENL x N^2) is badly wrong because GRD is posted at 10 m
    # from ~20 m resolution, so neighbouring pixels are correlated. Estimating
    # on both the raw and multi-looked stacks and taking the *ratio* cancels
    # the correlation bias that affects both equally, then anchors the result
    # to the product's known nominal ENL.
    enl_raw_est = omnibus_np.estimate_enl(stack[:3])
    if args.multilook > 1:
        stack = [omnibus_np.multilook(im, args.multilook) for im in stack]
    pixel_deg = args.pixel_deg * max(1, args.multilook)
    h, w = stack[0].shape[1:]

    if args.enl is not None:
        enl = args.enl
        enl_note = 'user supplied'
    elif args.multilook > 1:
        enl_ml_est = omnibus_np.estimate_enl(stack[:3])
        enl = omnibus_np.ENL * (enl_ml_est / enl_raw_est)
        enl_note = ('estimated: raw %.2f, multi-looked %.2f, gain %.2fx'
                    % (enl_raw_est, enl_ml_est, enl_ml_est / enl_raw_est))
    else:
        enl = omnibus_np.ENL
        enl_note = 'nominal for IW GRDH'
    print('multilook %dx -> %d x %d px at %.5f deg | ENL %.2f (%s)'
          % (args.multilook, h, w, pixel_deg, enl, enl_note), file=sys.stderr)

    print('running the omnibus test...', file=sys.stderr)
    res = tiled_change_maps(stack, args.alpha, args.median, enl, args.tile)

    first_neg = first_negative_interval(res['bmap'])
    detections = first_neg > 0

    classes = worldcover.read_aoi(bbox, pixel_deg)
    woody = worldcover.woody_mask(classes)
    clearing = detections & woody if args.woody_mask else detections

    # Pixel area from the local metre-per-degree scale.
    lat_mid = (bbox[1] + bbox[3]) / 2.0
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_mid))
    pixel_area_ha = (pixel_deg * m_per_deg_lat) * \
                    (pixel_deg * m_per_deg_lon) / 10000.0

    clearing = apply_mmu(clearing, args.min_mmu_ha, pixel_area_ha)
    first_neg_masked = np.where(clearing, first_neg, 0).astype(np.int16)

    per_interval = {}
    for i in range(1, len(stack)):
        n_px = int((first_neg_masked == i).sum())
        if n_px:
            per_interval[i] = {'from': dates[i - 1], 'to': dates[i],
                               'pixels': n_px,
                               'hectares': round(n_px * pixel_area_ha, 3)}

    summary = {
        'method': 'sequential omnibus test (Conradsen et al. 2016), run locally',
        'data': 'ESA Sentinel-1 IW GRDH via the public AWS mirror, no Earth Engine',
        'bbox': list(bbox),
        'relative_orbit': track,
        'n_scenes': len(stack),
        'dates': dates,
        'grid': {'height': h, 'width': w, 'pixel_deg': pixel_deg,
                 'source_pixel_deg': args.pixel_deg,
                 'multilook': args.multilook,
                 'pixel_area_ha': round(pixel_area_ha, 6)},
        'alpha': args.alpha,
        'enl': round(enl, 3),
        'enl_note': enl_note,
        'median_filter': args.median,
        'woody_mask': args.woody_mask,
        'min_mmu_ha': args.min_mmu_ha,
        'landcover_fractions': {k: round(v, 4)
                                for k, v in worldcover.summarise(classes).items()},
        'geolocation_fit_px': diags[0]['vv'] if diags else None,
        'aoi_area_ha': round(h * w * pixel_area_ha, 1),
        'woody_area_ha': round(int(woody.sum()) * pixel_area_ha, 1),
        'detected_clearing_ha': round(int(clearing.sum()) * pixel_area_ha, 3),
        'negative_change_before_mask_ha': round(int(detections.sum()) * pixel_area_ha, 3),
        'per_interval': per_interval,
    }

    with open(os.path.join(args.outdir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    write_geotiff(os.path.join(args.outdir, 'clearing.tif'),
                  [first_neg_masked, res['fmap'], classes.astype(np.int16)],
                  ['first_negative_interval', 'n_changes', 'worldcover'],
                  bbox, pixel_deg)
    np.savez_compressed(os.path.join(args.outdir, 'arrays.npz'),
                        first_neg=first_neg_masked, fmap=res['fmap'],
                        classes=classes, woody=woody,
                        vv_first=stack[0][0], vv_last=stack[-1][0],
                        vh_first=stack[0][1], vh_last=stack[-1][1])

    print(json.dumps({k: summary[k] for k in
                      ('n_scenes', 'relative_orbit', 'aoi_area_ha',
                       'woody_area_ha', 'detected_clearing_ha')}, indent=2))
    print('wrote %s' % args.outdir, file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
