#!/usr/bin/env python3
"""Screen radar candidates with fused local contrast and optical evidence.

    python -m land_clearing.local_s1.confirm rlx_cobar_town rlx_cobar_e \
        --outdir shapefiles --name nsw_clearing_2023_fused

This is stage two of a two-stage design. Stage one (``run.py``) uses a
deliberately permissive radar threshold so few real events are missed; this
stage does the rejecting, using evidence the omnibus test cannot see: how the
candidate compares to its own surroundings, and whether Sentinel-2 agrees.

Running the radar loose and the fusion tight is the point. Tightening the radar
alone trades recall for precision one-for-one; adding an independent sensor
buys precision without paying recall, because the two error modes are
different -- speckle is uncorrelated with cloud, and soil moisture moves radar
far more than it moves NDVI.
"""

import argparse
import json
import os
import sys

import numpy as np

from . import fusion, vectorise

# Optical composites are shared between candidates falling in the same cell
# with the same event window.
CELL_DEG = 0.02
CHIP_HALF = 0.03


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rundirs', nargs='+')
    p.add_argument('--outdir', default='shapefiles')
    p.add_argument('--name', default='nsw_clearing_fused')
    p.add_argument('--max-scenes', type=int, default=4,
                   help='S2 scenes composited per side (default 4)')
    p.add_argument('--vh-contrast', type=float, default=fusion.VH_CONTRAST_DB)
    p.add_argument('--ndvi-contrast', type=float, default=fusion.NDVI_CONTRAST)
    p.add_argument('--min-frac', type=float, default=fusion.MIN_STRONG_FRACTION,
                   help='minimum share of a polygon that must clear the radar '
                        'contrast threshold (default 0.35)')
    p.add_argument('--skip-optical', action='store_true')
    a = p.parse_args(argv)

    import geopandas as gpd
    import pandas as pd
    from rasterio.transform import from_origin

    from .reader import _prepare_gdal_env
    _prepare_gdal_env()
    os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tiff,.tif,.jp2'
    from . import sentinel2 as s2

    frames = []
    granule_cache = {}
    composite_cache = {}
    for rd in a.rundirs:
        gdf, summary = vectorise.build(rd)
        if gdf.empty:
            print('%s: no candidates' % rd, file=sys.stderr)
            continue
        z = np.load(os.path.join(rd, 'arrays.npz'))
        if 'stack' not in z.files:
            raise SystemExit('%s has no saved stack; re-run run.py' % rd)
        stack = z['stack'].astype(np.float64)
        woody = z['woody']
        dates = summary['dates']
        bbox = summary['bbox']
        pdg = summary['grid']['pixel_deg']
        transform = from_origin(bbox[0], bbox[3], pdg, pdg)
        shape = stack.shape[2:]
        print('%s: %d candidates' % (rd, len(gdf)), file=sys.stderr)

        rows = []
        for _, r in gdf.iterrows():
            poly = fusion.rasterize_polygon(r.geometry, transform, shape)
            ring = fusion.background_ring(poly, valid=woody)
            vh = fusion.radar_contrast(
                stack, dates, poly, ring, str(r['date_from'])[:10],
                str(r['date_to'])[:10], threshold=a.vh_contrast)
            out = {'vh_poly': _r(vh['poly']), 'vh_bkg': _r(vh['bkg']),
                   'vh_contr': _r(vh['mean']), 'vh_core': _r(vh['core']),
                   'vh_frac': _r(vh['frac'], 3)}

            if not a.skip_optical:
                lon, lat = float(r['lon']), float(r['lat'])
                zone, band, sq = s2.mgrs_tile(lon, lat)
                key = (zone, band, sq)
                if key not in granule_cache:
                    granule_cache[key] = s2.list_granules(
                        zone, band, sq, '2023-01-01', '2024-02-28')
                gran = granule_cache[key]
                bcut = str(r['date_from'])[:10].replace('-', '')
                acut = str(r['date_to'])[:10].replace('-', '')

                # Candidates cluster, and an optical composite is by far the
                # most expensive thing here, so share one across neighbours
                # on the same event window. The chip is centred on a CELL of
                # CELL_DEG rather than on the polygon, and is half again as
                # wide, so any polygon in the cell is comfortably inside it.
                cell = (round(lon / CELL_DEG), round(lat / CELL_DEG),
                        bcut, acut)
                if cell not in composite_cache:
                    clon = round(lon / CELL_DEG) * CELL_DEG
                    clat = round(lat / CELL_DEG) * CELL_DEG
                    chip = (clon - CHIP_HALF, clat - CHIP_HALF,
                            clon + CHIP_HALF, clat + CHIP_HALF)
                    before = list(reversed([g for g in gran if g[0] <= bcut]))
                    after = [g for g in gran if g[0] >= acut]
                    composite_cache[cell] = (
                        chip,
                        s2.ndvi_composite(chip, before, max_scenes=a.max_scenes),
                        s2.ndvi_composite(chip, after, max_scenes=a.max_scenes))
                chip, pre_c, post_c = composite_cache[cell]
                oc = fusion.contrast_from_composites(r.geometry, pre_c, post_c)
                if oc:
                    out.update({'ndvi_poly': _r(oc['ndvi_poly'], 4),
                                'ndvi_bkg': _r(oc['ndvi_bkg'], 4),
                                'ndvi_contr': _r(oc['ndvi_contrast'], 4),
                                's2_n_pre': oc['n_pre'],
                                's2_n_post': oc['n_post']})
                else:
                    out.update({'ndvi_poly': None, 'ndvi_bkg': None,
                                'ndvi_contr': None, 's2_n_pre': 0,
                                's2_n_post': 0})
            out['verdict'] = fusion.verdict(
                vh, out.get('ndvi_contr'), a.vh_contrast, a.ndvi_contrast,
                a.min_frac)
            print('   id %-3s %6.1f ha  mean %6s core %6s frac %5s  '
                  'ndvi %8s  -> %s'
                  % (r['clear_id'], r['area_ha'], out['vh_contr'],
                     out['vh_core'], out['vh_frac'], out.get('ndvi_contr'),
                     out['verdict']), file=sys.stderr)
            rows.append(out)

        for k in rows[0]:
            gdf[k] = [row.get(k) for row in rows]
        frames.append(gdf)

    if not frames:
        raise SystemExit('No candidates in any run directory.')
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                              crs='EPSG:4326')
    merged = merged.sort_values('area_ha', ascending=False).reset_index(drop=True)
    merged['clear_id'] = np.arange(1, len(merged) + 1)

    counts = merged['verdict'].value_counts().to_dict()
    print('\nverdicts: %s' % counts, file=sys.stderr)
    conf = merged[merged['verdict'] == 'confirmed']
    print('confirmed: %d polygons, %.1f ha' % (len(conf), conf['area_ha'].sum()),
          file=sys.stderr)

    for path in vectorise.write_outputs(merged, a.outdir, a.name):
        print(path)
    if len(conf):
        for path in vectorise.write_outputs(conf.reset_index(drop=True),
                                            a.outdir, a.name + '_confirmed'):
            print(path)
    return 0


def _r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


if __name__ == '__main__':
    raise SystemExit(main())
