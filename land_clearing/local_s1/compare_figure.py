#!/usr/bin/env python3
"""Side-by-side Sentinel-2 and Sentinel-1 before/after of detected clearing.

    python -m land_clearing.local_s1.compare_figure fin_cobar_town \
        --lon 145.79195 --lat -31.54282 --polygons shapefiles/nsw_clearing_2023.geojson

Top row is Sentinel-2 L2A true colour either side of the event plus the NDVI
change; bottom row is the Sentinel-1 VH backscatter the detection was actually
made from. Both difference panels use one diverging scheme with a neutral
midpoint, brown for loss, so the same legend reads for both even though the
units differ (NDVI vs dB).

The Sentinel-2 pair is chosen by cloud cover measured over this chip rather
than over the whole granule -- a scene can be 40% cloudy and perfectly clear
across 3 km.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

# Diverging: two hues, neutral midpoint, brown = loss for both NDVI and dB.
DIVERGING = 'BrBG'
POLY_COLOUR = '#FF1FA0'
INK = '#1a1a1a'
MUTED = '#666666'


def _db(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.where(x > 0, x, np.nan))


def _stretch(bands, lo=2, hi=98):
    """Joint percentile stretch across a list of same-band arrays."""
    allv = np.concatenate([b[np.isfinite(b)].ravel() for b in bands])
    return np.percentile(allv, lo), np.percentile(allv, hi)


def _rgb(chip, limits):
    out = np.dstack([
        np.clip((chip[k] - limits[k][0]) / (limits[k][1] - limits[k][0]), 0, 1)
        for k in ('red', 'green', 'blue')])
    return out


def _crop_s1(z, summary, extent):
    """Crop the S1 stack to a lon/lat extent. Returns (stack, extent_used)."""
    bbox = summary['bbox']
    pd = summary['grid']['pixel_deg']
    lon0, lon1, lat0, lat1 = extent
    c0 = max(0, int(round((lon0 - bbox[0]) / pd)))
    c1 = int(round((lon1 - bbox[0]) / pd))
    r0 = max(0, int(round((bbox[3] - lat1) / pd)))
    r1 = int(round((bbox[3] - lat0) / pd))
    stack = z['stack'][:, :, r0:r1, c0:c1]
    used = [bbox[0] + c0 * pd, bbox[0] + c1 * pd,
            bbox[3] - r1 * pd, bbox[3] - r0 * pd]
    return stack, used


def build(rundir, lon, lat, half_deg, polygons, outpath, before_cut, after_cut,
          window=4, max_cloud=0.02, search_limit=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    from .reader import _prepare_gdal_env
    _prepare_gdal_env()
    os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tiff,.tif,.jp2'
    from . import sentinel2 as s2

    bbox = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
    summary = json.load(open(os.path.join(rundir, 'summary.json')))
    z = np.load(os.path.join(rundir, 'arrays.npz'))
    dates = summary['dates']

    zone, band, square = s2.mgrs_tile(lon, lat)
    print('MGRS tile %d%s%s' % (zone, band, square), file=sys.stderr)
    # Derive the search range from the event dates. A hardcoded range silently
    # empties the "before" list for any event earlier than its start, which
    # then surfaces as "no cloud-free pair" rather than as the real cause.
    lo = (datetime.date.fromisoformat(_iso(before_cut))
          - datetime.timedelta(days=150)).isoformat()
    hi = (datetime.date.fromisoformat(_iso(after_cut))
          + datetime.timedelta(days=150)).isoformat()
    gran = s2.list_granules(zone, band, square, lo, hi)
    print('%d granules between %s and %s' % (len(gran), lo, hi), file=sys.stderr)

    print('Sentinel-2 before:', file=sys.stderr)
    d_pre, p_pre = s2.find_cloudfree(
        bbox, list(reversed([g for g in gran if g[0] <= before_cut])),
        max_cloud=max_cloud, limit=search_limit)
    print('Sentinel-2 after:', file=sys.stderr)
    d_post, p_post = s2.find_cloudfree(
        bbox, [g for g in gran if g[0] >= after_cut],
        max_cloud=max_cloud, limit=search_limit)
    # Fall back to the least cloudy scene rather than producing nothing: over
    # Pilliga in early May no scene before the event is anywhere near clear,
    # though the fusion stage still measured it from partially valid
    # composites.
    cloud_pre = cloud_post = 0.0
    if not p_pre:
        d_pre, p_pre, cloud_pre = s2.find_least_cloudy(
            bbox, list(reversed([g for g in gran if g[0] <= before_cut])),
            limit=search_limit)
    if not p_post:
        d_post, p_post, cloud_post = s2.find_least_cloudy(
            bbox, [g for g in gran if g[0] >= after_cut], limit=search_limit)
    if not (p_pre and p_post):
        raise SystemExit('No usable Sentinel-2 pair over this chip at all.')
    suffix_pre = '' if not cloud_pre else ' (%.0f%% cloud)' % (100 * cloud_pre)
    suffix_post = '' if not cloud_post else ' (%.0f%% cloud)' % (100 * cloud_post)

    chip_pre = s2.read_rgb(p_pre, bbox)
    chip_post = s2.read_rgb(p_post, bbox)
    extent = chip_pre['extent']

    limits = {k: _stretch([chip_pre[k], chip_post[k]])
              for k in ('red', 'green', 'blue')}
    ndvi = lambda c: (c['nir'] - c['red']) / (c['nir'] + c['red'] + 1e-9)
    dndvi = ndvi(chip_post) - ndvi(chip_pre)

    stack, s1_extent = _crop_s1(z, summary, extent)
    i_pre = [i for i, d in enumerate(dates) if d <= before_cut[:4] + '-'
             + before_cut[4:6] + '-' + before_cut[6:]][-window:]
    i_post = [i for i, d in enumerate(dates) if d >= after_cut[:4] + '-'
              + after_cut[4:6] + '-' + after_cut[6:]][:window]
    vh_pre = np.nanmean(_db(stack[i_pre, 1].astype(np.float64)), axis=0)
    vh_post = np.nanmean(_db(stack[i_post, 1].astype(np.float64)), axis=0)
    dvh = vh_post - vh_pre

    import geopandas as gpd
    from shapely.geometry import box as sbox
    gdf = gpd.read_file(polygons)
    gdf = gdf[gdf.intersects(sbox(*bbox))]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.4))
    vlim_ndvi = float(np.nanpercentile(np.abs(dndvi), 99))
    vlim_vh = float(np.nanpercentile(np.abs(dvh), 99))

    panels = [
        (axes[0, 0], _rgb(chip_pre, limits), None, extent,
         'Sentinel-2 true colour\n%s (before)%s' % (_fmt(d_pre), suffix_pre)),
        (axes[0, 1], _rgb(chip_post, limits), None, extent,
         'Sentinel-2 true colour\n%s (after)%s' % (_fmt(d_post), suffix_post)),
        (axes[0, 2], dndvi, ('NDVI change', vlim_ndvi), extent,
         'NDVI change\n%s to %s' % (_fmt(d_pre), _fmt(d_post))),
        (axes[1, 0], vh_pre, None, s1_extent,
         'Sentinel-1 VH $\\sigma^0$\nmean of %d before' % len(i_pre)),
        (axes[1, 1], vh_post, None, s1_extent,
         'Sentinel-1 VH $\\sigma^0$\nmean of %d after' % len(i_post)),
        (axes[1, 2], dvh, ('VH change (dB)', vlim_vh), s1_extent,
         'VH backscatter change\nthis is what the detector sees'),
    ]

    for ax, data, div, ext, title in panels:
        if div is None:
            if data.ndim == 3:
                ax.imshow(data, extent=ext, origin='upper')
            else:
                ax.imshow(data, extent=ext, origin='upper', cmap='gray',
                          vmin=np.nanpercentile(vh_pre, 2),
                          vmax=np.nanpercentile(vh_pre, 98))
        else:
            label, vlim = div
            im = ax.imshow(data, extent=ext, origin='upper', cmap=DIVERGING,
                           vmin=-vlim, vmax=vlim)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(label, fontsize=9, color=INK)
            cb.ax.tick_params(labelsize=8, colors=MUTED)
            cb.outline.set_visible(False)
        gdf.boundary.plot(ax=ax, color=POLY_COLOUR, linewidth=1.6,
                          path_effects=[pe.withStroke(linewidth=3.2,
                                                      foreground='white')])
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.tick_params(labelsize=7.5, colors=MUTED)
        for s in ax.spines.values():
            s.set_color('#dddddd')

    handles = [Line2D([0], [0], color=POLY_COLOUR, lw=1.8,
                      label='detected clearing (%d polygon%s, %.1f ha)'
                            % (len(gdf), '' if len(gdf) == 1 else 's',
                               gdf['area_ha'].sum()))]
    fig.legend(handles=handles, loc='lower center', ncol=1, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.018))
    fig.suptitle('Land clearing near Cobar, NSW — optical and radar, before and after\n'
                 'Sentinel-1 relative orbit %d; the event falls between %s and %s'
                 % (summary['relative_orbit'], _fmt(before_cut), _fmt(after_cut)),
                 fontsize=13.5, color=INK)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    fig.savefig(outpath, dpi=140)
    return outpath, d_pre, d_post


def _iso(d):
    return '%s-%s-%s' % (d[:4], d[4:6], d[6:]) if len(d) == 8 else d[:10]


def _fmt(d):
    return '%s-%s-%s' % (d[:4], d[4:6], d[6:]) if len(d) == 8 else d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rundir')
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--lat', type=float, required=True)
    p.add_argument('--half-deg', type=float, default=0.016)
    p.add_argument('--polygons', required=True)
    p.add_argument('--before', default='20230913', help='last date before event')
    p.add_argument('--after', default='20231007', help='first date after event')
    p.add_argument('--max-cloud', type=float, default=0.02,
                   help='cloud fraction tolerated over the chip (default 0.02). '
                        'Raise it for winter windows -- around Cobar in July '
                        'whole runs of dates are fully clouded.')
    p.add_argument('--search-limit', type=int, default=12,
                   help='granules probed per side (default 12)')
    p.add_argument('--out', default='compare.png')
    a = p.parse_args(argv)
    out, d0, d1 = build(a.rundir, a.lon, a.lat, a.half_deg, a.polygons, a.out,
                        a.before, a.after, max_cloud=a.max_cloud,
                        search_limit=a.search_limit)
    print('%s  (S2 %s -> %s)' % (out, d0, d1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
