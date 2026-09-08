#!/usr/bin/env python3
"""Confirm radar change candidates using local contrast and optical evidence.

The omnibus test asks whether a pixel changed relative to its own history. That
is the right question statistically and the wrong one operationally, because a
rain front changes every pixel's history at once. Over the Cobar runs the four
false positives were all area-wide backscatter drops; the one true detection was
a *localised* drop.

So this stage asks a different question of each candidate: did it change more
than its own surroundings did? Two independent answers are computed.

``vh_contrast``
    Sentinel-1 VH drop inside the polygon minus the drop in a ring of woody
    vegetation around it. An area-wide moisture change cancels; localised
    clearing does not. On the five known polygons this ranked them correctly
    where the AOI-wide ``rain_flag`` did not.

``ndvi_contrast``
    The same difference computed on median cloud-free Sentinel-2 NDVI
    composites either side of the event. Independent of the radar entirely, so
    agreement between the two is genuine corroboration rather than the same
    error twice.

Compositing several optical scenes per side, rather than picking one clear
date, matters: a single scene still carries that day's soil moisture and sun
angle, which is exactly the nuisance being controlled for.
"""

import numpy as np

# Ring geometry, in pixels of whichever grid is being used. The inner gap keeps
# the polygon's own edge pixels out of its background estimate.
RING_INNER = 5
RING_OUTER = 25
S2_SCALE = 8            # S2 is 10 m against the 89 m radar grid

# Thresholds. Deliberately looser than the observed true positive so the
# decision is not fitted to a single event.
VH_CONTRAST_DB = -1.5
NDVI_CONTRAST = -0.03


def rasterize_polygon(geom, transform, shape):
    from rasterio.features import rasterize
    return rasterize([(geom, 1)], out_shape=shape, transform=transform,
                     dtype='uint8').astype(bool)


def background_ring(poly, valid=None, inner=RING_INNER, outer=RING_OUTER):
    """Annulus around a polygon, optionally restricted to valid ground."""
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(poly, iterations=outer) & ~binary_dilation(
        poly, iterations=inner)
    if valid is not None:
        ring = ring & valid
    return ring


def _mean(stack_db, idx, mask):
    if not len(idx) or not mask.any():
        return np.nan
    return float(np.nanmean(stack_db[idx][:, mask]))


def radar_contrast(stack, dates, poly, ring, before, after, band=1, window=4):
    """VH (or VV) change inside the polygon, in its background, and the difference."""
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10.0 * np.log10(np.where(stack[:, band] > 0, stack[:, band], np.nan))
    ip = [i for i, d in enumerate(dates) if d <= before][-window:]
    iq = [i for i, d in enumerate(dates) if d >= after][:window]
    d_poly = _mean(db, iq, poly) - _mean(db, ip, poly)
    d_ring = _mean(db, iq, ring) - _mean(db, ip, ring)
    return d_poly, d_ring, d_poly - d_ring


def contrast_from_composites(geom, pre, post):
    """NDVI change inside a polygon vs its surroundings, from two composites.

    Split out from ``optical_contrast`` so one composite pair can serve every
    candidate near it -- building the composites is the expensive step, and
    measuring a polygon against them is nearly free.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    if pre is None or post is None:
        return None
    h = min(pre['ndvi'].shape[0], post['ndvi'].shape[0])
    w = min(pre['ndvi'].shape[1], post['ndvi'].shape[1])
    d = post['ndvi'][:h, :w] - pre['ndvi'][:h, :w]

    to_utm = Transformer.from_crs('EPSG:4326', pre['crs'], always_xy=True)
    geom_utm = shp_transform(lambda x, y, z=None: to_utm.transform(x, y), geom)
    poly = rasterize_polygon(geom_utm, pre['transform'], (h, w))
    if not poly.any():
        return None
    ring = background_ring(poly, valid=np.isfinite(d),
                           inner=RING_INNER * S2_SCALE,
                           outer=RING_OUTER * S2_SCALE)
    if not ring.any():
        return None
    with np.errstate(invalid='ignore'):
        d_poly = float(np.nanmean(d[poly]))
        d_ring = float(np.nanmean(d[ring]))
    return {'ndvi_poly': d_poly, 'ndvi_bkg': d_ring,
            'ndvi_contrast': d_poly - d_ring,
            'n_pre': len(pre['dates']), 'n_post': len(post['dates']),
            'pre_dates': pre['dates'], 'post_dates': post['dates']}


def optical_contrast(geom, bbox, before_granules, after_granules,
                     max_scenes=3):
    """Build composites for one polygon and measure its NDVI contrast."""
    from . import sentinel2 as s2
    return contrast_from_composites(
        geom,
        s2.ndvi_composite(bbox, before_granules, max_scenes=max_scenes),
        s2.ndvi_composite(bbox, after_granules, max_scenes=max_scenes))


def verdict(vh_contrast, ndvi_contrast,
            vh_thresh=VH_CONTRAST_DB, ndvi_thresh=NDVI_CONTRAST):
    """Combine the two contrasts into a label.

    ``confirmed``  both sensors agree the change is local
    ``radar_only`` radar says local, optical does not corroborate (or is absent)
    ``optical_only`` optical says local, radar does not
    ``rejected``   neither: the change is as large in the surroundings
    """
    r = vh_contrast is not None and np.isfinite(vh_contrast) and vh_contrast <= vh_thresh
    o = (ndvi_contrast is not None and np.isfinite(ndvi_contrast)
         and ndvi_contrast <= ndvi_thresh)
    if r and o:
        return 'confirmed'
    if r:
        return 'radar_only'
    if o:
        return 'optical_only'
    return 'rejected'
