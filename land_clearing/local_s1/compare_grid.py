#!/usr/bin/env python3
"""One figure covering every detection: optical and radar, before and after.

    python -m land_clearing.local_s1.compare_grid \
        --polygons shapefiles/nsw_clearing_2023.geojson \
        --runs cobar_town=fin_cobar_town cobar_e=fin_cobar_e --out grid.png

One row per detection site, four columns: Sentinel-2 true colour either side of
that site's own event window, the NDVI change, and the Sentinel-1 VH change the
detection was actually made from.

Each row gets its own Sentinel-2 pair, chosen from that polygon's own break
interval -- the detections span July to October, so a single date pair would
straddle the wrong window for most of them.

Polygons closer than ``CLUSTER_DEG`` share a chip rather than getting a row
each, since two detections 300 m apart in the same paddock are one site.
"""

import argparse
import json
import os
import sys

import numpy as np

DIVERGING = 'BrBG'
POLY_COLOUR = '#FF1FA0'
INK = '#1a1a1a'
MUTED = '#666666'
CLUSTER_DEG = 0.01
HALF_DEG = 0.014


def _db(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.where(x > 0, x, np.nan))


def cluster(gdf):
    """Group polygons whose centroids fall within CLUSTER_DEG."""
    sites, used = [], set()
    rows = list(gdf.iterrows())
    for i, (_, a) in enumerate(rows):
        if i in used:
            continue
        members = [i]
        used.add(i)
        for j, (_, b) in enumerate(rows):
            if j in used:
                continue
            if (abs(a['lon'] - b['lon']) < CLUSTER_DEG
                    and abs(a['lat'] - b['lat']) < CLUSTER_DEG):
                members.append(j)
                used.add(j)
        sub = gdf.iloc[members]
        sites.append({
            'idx': members,
            'lon': float(sub['lon'].mean()),
            'lat': float(sub['lat'].mean()),
            'area_ha': float(sub['area_ha'].sum()),
            'aoi': sub.iloc[0]['aoi'],
            # Widest window the members span, so the chip brackets them all.
            'before': min(sub['date_from']),
            'after': max(sub['date_to']),
            'flag': 'check' if (sub['rain_flag'] == 'check').any() else 'ok',
            'ids': list(sub['clear_id']),
        })
    return sorted(sites, key=lambda s: -s['area_ha'])


def crop_s1(z, summary, extent):
    bbox = summary['bbox']
    pd = summary['grid']['pixel_deg']
    lon0, lon1, lat0, lat1 = extent
    c0 = max(0, int(round((lon0 - bbox[0]) / pd)))
    c1 = min(z['stack'].shape[3], int(round((lon1 - bbox[0]) / pd)))
    r0 = max(0, int(round((bbox[3] - lat1) / pd)))
    r1 = min(z['stack'].shape[2], int(round((bbox[3] - lat0) / pd)))
    used = [bbox[0] + c0 * pd, bbox[0] + c1 * pd,
            bbox[3] - r1 * pd, bbox[3] - r0 * pd]
    return z['stack'][:, :, r0:r1, c0:c1], used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--polygons', required=True)
    p.add_argument('--runs', nargs='+', required=True,
                   help='aoi=rundir pairs')
    p.add_argument('--out', default='compare_grid.png')
    a = p.parse_args(argv)
    runs = dict(r.split('=', 1) for r in a.runs)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    import geopandas as gpd
    from shapely.geometry import box as sbox

    from .reader import _prepare_gdal_env
    _prepare_gdal_env()
    os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tiff,.tif,.jp2'
    from . import sentinel2 as s2

    gdf = gpd.read_file(a.polygons)
    # The GeoJSON driver infers date fields into Timestamps; the rest of this
    # module compares and slices them as ISO strings.
    for col in ('date_from', 'date_to'):
        gdf[col] = gdf[col].astype(str).str.slice(0, 10)
    sites = cluster(gdf)
    print('%d polygons -> %d sites' % (len(gdf), len(sites)), file=sys.stderr)

    panels = []
    for site in sites:
        lon, lat = site['lon'], site['lat']
        bbox = (lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG)
        zone, band, sq = s2.mgrs_tile(lon, lat)
        gran = s2.list_granules(zone, band, sq, '2023-04-01', '2023-12-31')
        bcut = site['before'].replace('-', '')
        acut = site['after'].replace('-', '')
        print('site %s %.1f ha, tile %d%s%s, window %s..%s'
              % (site['ids'], site['area_ha'], zone, band, sq, bcut, acut),
              file=sys.stderr)
        d0, p0 = s2.find_cloudfree(
            bbox, list(reversed([g for g in gran if g[0] <= bcut])), verbose=False)
        d1, p1 = s2.find_cloudfree(
            bbox, [g for g in gran if g[0] >= acut], verbose=False)
        if not (p0 and p1):
            print('  no cloud-free pair, skipping', file=sys.stderr)
            continue
        pre, post = s2.read_rgb(p0, bbox), s2.read_rgb(p1, bbox)
        extent = pre['extent']

        rundir = runs[site['aoi']]
        summary = json.load(open(os.path.join(rundir, 'summary.json')))
        z = np.load(os.path.join(rundir, 'arrays.npz'))
        dates = summary['dates']
        stack, s1_extent = crop_s1(z, summary, extent)
        ip = [i for i, d in enumerate(dates) if d <= site['before']][-4:]
        iq = [i for i, d in enumerate(dates) if d >= site['after']][:4]
        vh_pre = np.nanmean(_db(stack[ip, 1].astype(np.float64)), axis=0)
        dvh = np.nanmean(_db(stack[iq, 1].astype(np.float64)), axis=0) - vh_pre

        ndvi = lambda c: (c['nir'] - c['red']) / (c['nir'] + c['red'] + 1e-9)
        panels.append((site, pre, post, ndvi(post) - ndvi(pre), dvh,
                       extent, s1_extent, d0, d1,
                       gdf[gdf.intersects(sbox(*bbox))]))

    n = len(panels)
    fig, axes = plt.subplots(n, 4, figsize=(17.5, 4.35 * n), squeeze=False)

    for r, (site, pre, post, dndvi, dvh, ext, s1ext, d0, d1, sub) in enumerate(panels):
        lims = {k: (np.percentile(np.concatenate([pre[k].ravel(), post[k].ravel()]), 2),
                    np.percentile(np.concatenate([pre[k].ravel(), post[k].ravel()]), 98))
                for k in ('red', 'green', 'blue')}
        rgb = lambda c: np.dstack([np.clip((c[k] - lims[k][0])
                                           / (lims[k][1] - lims[k][0]), 0, 1)
                                   for k in ('red', 'green', 'blue')])
        vn = float(np.nanpercentile(np.abs(dndvi), 99))
        vv = float(np.nanpercentile(np.abs(dvh), 99)) or 1.0

        cells = [
            (rgb(pre), None, ext, 'S2 %s (before)' % _f(d0)),
            (rgb(post), None, ext, 'S2 %s (after)' % _f(d1)),
            (dndvi, ('NDVI change', vn), ext, 'NDVI change'),
            (dvh, ('VH change (dB)', vv), s1ext, 'S1 VH change'),
        ]
        for c, (data, div, e, title) in enumerate(cells):
            ax = axes[r][c]
            if div is None:
                ax.imshow(data, extent=e, origin='upper')
            else:
                label, vlim = div
                im = ax.imshow(data, extent=e, origin='upper', cmap=DIVERGING,
                               vmin=-vlim, vmax=vlim)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label(label, fontsize=8, color=INK)
                cb.ax.tick_params(labelsize=7, colors=MUTED)
                cb.outline.set_visible(False)
            sub.boundary.plot(ax=ax, color=POLY_COLOUR, linewidth=1.5,
                              path_effects=[pe.withStroke(linewidth=3,
                                                          foreground='white')])
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(title, fontsize=9.5, color=INK)
            ax.tick_params(labelsize=6.5, colors=MUTED)
            for s in ax.spines.values():
                s.set_color('#dddddd')

        flag = ('  ⚠ rain_flag=check' if site['flag'] == 'check' else '')
        axes[r][0].set_ylabel(
            'polygon %s — %.1f ha\n%s to %s%s'
            % (','.join(str(i) for i in site['ids']), site['area_ha'],
               site['before'], site['after'], flag),
            fontsize=9.5, color=INK, labelpad=8)

    handles = [Line2D([0], [0], color=POLY_COLOUR, lw=1.8,
                      label='detected clearing')]
    fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle('Every Sentinel-1 clearing detection, NSW 2023 — optical and radar, before and after\n'
                 'each row uses the Sentinel-2 pair bracketing that detection\'s own event window',
                 fontsize=13.5, color=INK)
    fig.tight_layout(rect=[0, 0.022, 1, 1 - 0.055 / n])
    fig.savefig(a.out, dpi=125)
    print(a.out)
    return 0


def _f(d):
    return '%s-%s-%s' % (d[:4], d[4:6], d[6:]) if len(d) == 8 else d


if __name__ == '__main__':
    raise SystemExit(main())
