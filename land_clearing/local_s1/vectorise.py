#!/usr/bin/env python3
"""Turn a detection raster into clearing polygons with attributes.

    python -m land_clearing.local_s1.vectorise fin_cobar_town --outdir shapefiles

Writes an ESRI Shapefile (zipped, with .prj), a GeoPackage and GeoJSON. The
shapefile is there because it was asked for; prefer the GeoPackage if your
software will take it, since shapefiles truncate field names to 10 characters,
have no real date type and split one layer across five files.

Two attributes matter more than the geometry when triaging results:

``persist``
    Fraction of acquisitions after the break that stay at least 1 dB below the
    pre-break mean. Clearing is permanent, so real events sit near 1.0. A
    transient dip from soil moisture or a wet-to-dry transition falls back and
    scores low. This is the single most useful column for separating the two.

``rain_flag``
    Whether the AOI-wide woody median *also* moved sharply in the same
    interval. A basin-wide moisture change is a real backscatter change and the
    detector is right to flag it, but it is not clearing. ``aoi_dlevel`` carries
    the underlying number in dB so you can set your own threshold.

Areas are computed in EPSG:3577 (GDA94 / Australian Albers), the equal-area
projection for national statistics; geometry is written in EPSG:4326.
"""

import argparse
import json
import os
import shutil
import sys
import zipfile

import numpy as np

# Shapefile DBF field names are capped at 10 characters, so every name here is
# already short enough to survive the write without silent truncation.
AREA_CRS = 'EPSG:3577'
PERSIST_DROP_DB = 1.0
RAIN_LEVEL_DB = 1.0
WINDOW = 4


def _db(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.where(x > 0, x, np.nan))


def _window_mean_db(stack_db, idx, mask):
    """Mean dB over a set of dates for the pixels in ``mask``."""
    if not len(idx):
        return np.nan
    return float(np.nanmean(stack_db[idx][:, mask]))


def build(rundir, min_area_ha=0.0):
    """Return a GeoDataFrame of clearing polygons for one run directory."""
    import geopandas as gpd
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from scipy.ndimage import label
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    summary = json.load(open(os.path.join(rundir, 'summary.json')))
    z = np.load(os.path.join(rundir, 'arrays.npz'))

    first_neg = z['first_neg']
    detected = first_neg > 0
    if not detected.any():
        return gpd.GeoDataFrame(
            [], columns=['geometry'], geometry='geometry', crs='EPSG:4326'), summary

    bbox = summary['bbox']
    pixel_deg = summary['grid']['pixel_deg']
    dates = summary['dates']
    transform = from_origin(bbox[0], bbox[3], pixel_deg, pixel_deg)

    has_stack = 'stack' in z.files
    if has_stack:
        stack = z['stack'].astype(np.float64)          # (T, 2, H, W)
        vv_db, vh_db = _db(stack[:, 0]), _db(stack[:, 1])
        woody = z['woody']
        # AOI-wide woody level per date, for the rain flag.
        level = np.array([np.nanmedian(vh_db[t][woody]) for t in range(len(dates))])
    else:
        vv_db = vh_db = level = None

    classes = z['classes']
    fmap = z['fmap']

    lab, n = label(detected, structure=np.ones((3, 3)))
    geoms = {}
    # connectivity=8 must match the 8-connected labelling above. With the
    # rasterio default of 4, a component whose parts touch only diagonally is
    # split into several polygons, and keeping the largest silently drops area.
    for geom, value in shapes(lab.astype(np.int32), mask=detected,
                              transform=transform, connectivity=8):
        v = int(value)
        if v:
            geoms.setdefault(v, []).append(shapely_shape(geom))

    from . import worldcover

    rows, polys = [], []
    for v, parts in sorted(geoms.items()):
        # Keep every part, as a MultiPolygon if need be, so the reported area
        # is the whole detection rather than its largest fragment.
        poly = parts[0] if len(parts) == 1 else unary_union(parts)
        mask = lab == v
        n_px = int(mask.sum())
        interval = int(np.median(first_neg[mask]))
        i0, i1 = interval - 1, interval          # dates[i0] -> dates[i1]

        row = {
            'clear_id': v,
            'aoi': os.path.basename(rundir.rstrip('/')).replace('fin_', ''),
            'n_pixels': n_px,
            'interval': interval,
            'date_from': dates[i0],
            'date_to': dates[i1],
            'n_changes': int(np.median(fmap[mask])),
        }

        if has_stack:
            pre = list(range(max(0, i0 - WINDOW + 1), i0 + 1))
            post = list(range(i1, min(len(dates), i1 + WINDOW)))
            vh_pre = _window_mean_db(vh_db, pre, mask)
            vh_post = _window_mean_db(vh_db, post, mask)
            vv_pre = _window_mean_db(vv_db, pre, mask)
            vv_post = _window_mean_db(vv_db, post, mask)

            after = np.arange(i1, len(dates))
            series_after = np.nanmean(vh_db[after][:, mask], axis=1)
            persist = float(np.mean(series_after <= vh_pre - PERSIST_DROP_DB))

            d_level = float(level[i1] - level[i0])
            row.update({
                'vh_pre': round(vh_pre, 2), 'vh_post': round(vh_post, 2),
                'vh_drop': round(vh_post - vh_pre, 2),
                'vv_pre': round(vv_pre, 2), 'vv_post': round(vv_post, 2),
                'vv_drop': round(vv_post - vv_pre, 2),
                'persist': round(persist, 3),
                'aoi_dlevel': round(d_level, 2),
                'rain_flag': 'check' if d_level < -RAIN_LEVEL_DB else 'ok',
            })

        cls_vals, cls_counts = np.unique(classes[mask], return_counts=True)
        top = int(cls_vals[cls_counts.argmax()])
        row['lc_class'] = top
        row['lc_name'] = worldcover.CLASS_NAMES.get(top, str(top))
        row['woody_frac'] = round(float(z['woody'][mask].mean()), 3)

        row.update({
            'orbit': summary['relative_orbit'],
            'n_scenes': summary['n_scenes'],
            'multilook': summary['grid'].get('multilook', 1),
            'enl': summary['enl'],
            'alpha': summary['alpha'],
        })
        rows.append(row)
        polys.append(poly)

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(rows, geometry=polys, crs='EPSG:4326')

    proj = gdf.to_crs(AREA_CRS)
    gdf['area_ha'] = (proj.area / 10000.0).round(3)
    # Polsby-Popper: 1.0 is a circle. Machine-cleared paddocks are blocky and
    # score moderately; speckle survivors are ragged and score low.
    gdf['compact'] = (4 * np.pi * proj.area / proj.length ** 2).round(3)
    cent = proj.centroid.to_crs('EPSG:4326')
    gdf['lon'] = cent.x.round(5)
    gdf['lat'] = cent.y.round(5)

    if min_area_ha:
        gdf = gdf[gdf['area_ha'] >= min_area_ha].copy()

    order = ['clear_id', 'aoi', 'area_ha', 'n_pixels', 'interval', 'date_from',
             'date_to', 'vh_pre', 'vh_post', 'vh_drop', 'vv_pre', 'vv_post',
             'vv_drop', 'persist', 'rain_flag', 'aoi_dlevel', 'n_changes',
             'compact', 'lc_class', 'lc_name', 'woody_frac', 'lon', 'lat',
             'orbit', 'n_scenes', 'multilook', 'enl', 'alpha', 'geometry']
    gdf = gdf[[c for c in order if c in gdf.columns]]
    return gdf.sort_values('area_ha', ascending=False).reset_index(drop=True), summary


def write_outputs(gdf, outdir, name):
    """Write shapefile (zipped), GeoPackage and GeoJSON. Returns paths."""
    os.makedirs(outdir, exist_ok=True)
    written = []

    gpkg = os.path.join(outdir, name + '.gpkg')
    gdf.to_file(gpkg, layer='clearing', driver='GPKG')
    written.append(gpkg)

    gj = os.path.join(outdir, name + '.geojson')
    gdf.to_file(gj, driver='GeoJSON')
    written.append(gj)

    shp_dir = os.path.join(outdir, '_shp_' + name)
    os.makedirs(shp_dir, exist_ok=True)
    shp = os.path.join(shp_dir, name + '.shp')
    gdf.to_file(shp, driver='ESRI Shapefile')
    zip_path = os.path.join(outdir, name + '_shapefile.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(shp_dir)):
            zf.write(os.path.join(shp_dir, f), f)
    shutil.rmtree(shp_dir)
    written.append(zip_path)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rundirs', nargs='+', help='run output directories')
    p.add_argument('--outdir', default='shapefiles')
    p.add_argument('--min-area-ha', type=float, default=0.0)
    p.add_argument('--name', default='nsw_clearing',
                   help='basename for the merged output')
    args = p.parse_args(argv)

    import geopandas as gpd
    frames = []
    for rd in args.rundirs:
        gdf, summary = build(rd, args.min_area_ha)
        if gdf.empty:
            print('%s: no detections' % rd, file=sys.stderr)
            continue
        print('%s: %d polygons, %.1f ha' % (rd, len(gdf), gdf['area_ha'].sum()),
              file=sys.stderr)
        frames.append(gdf)

    if not frames:
        raise SystemExit('No detections in any run directory.')
    merged = gpd.GeoDataFrame(
        __import__('pandas').concat(frames, ignore_index=True),
        crs='EPSG:4326').sort_values('area_ha', ascending=False).reset_index(drop=True)
    merged['clear_id'] = np.arange(1, len(merged) + 1)

    for path in write_outputs(merged, args.outdir, args.name):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
