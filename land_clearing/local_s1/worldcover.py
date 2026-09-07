"""ESA WorldCover 10 m land cover, read from its public S3 bucket.

Used for the same two jobs as in the Earth Engine pipelines: a baseline woody
mask (clearing needs something to clear) and cover strata. Tiles are 3x3 degree
COGs, so an AOI window is a cheap range read.
"""

import math

import numpy as np

BUCKET = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com'
VERSION = 'v200/2021'

TREE, SHRUB, GRASS, CROP = 10, 20, 30, 40
CLASS_NAMES = {10: 'tree cover', 20: 'shrubland', 30: 'grassland',
               40: 'cropland', 50: 'built-up', 60: 'bare/sparse',
               70: 'snow/ice', 80: 'water', 90: 'wetland', 95: 'mangrove',
               100: 'moss/lichen'}


def tile_name(lon, lat):
    """WorldCover tile covering a point, named by its south-west corner."""
    t_lon = int(math.floor(lon / 3.0) * 3)
    t_lat = int(math.floor(lat / 3.0) * 3)
    ns = 'N' if t_lat >= 0 else 'S'
    ew = 'E' if t_lon >= 0 else 'W'
    return 'ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
        ns, abs(t_lat), ew, abs(t_lon))


def read_aoi(bbox, pixel_deg):
    """Land cover class codes resampled onto the AOI grid (nearest neighbour)."""
    from scipy.ndimage import map_coordinates

    from .reader import _prepare_gdal_env, aoi_grid
    _prepare_gdal_env()
    import rasterio

    w, s, e, n = bbox
    name = tile_name((w + e) / 2.0, (s + n) / 2.0)
    url = f'/vsicurl/{BUCKET}/{VERSION}/map/{name}'

    lon_g, lat_g = aoi_grid(bbox, pixel_deg)
    with rasterio.open(url) as ds:
        # rasterio's ds.index() is scalar-only here, so invert the north-up
        # affine directly: col = (lon - c) / a, row = (lat - f) / e.
        a, _, c, _, e, f = ds.transform[:6]
        cols = np.floor((lon_g - c) / a).astype(int)
        rows = np.floor((lat_g - f) / e).astype(int)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        from rasterio.windows import Window
        block = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    return map_coordinates(block, [rows - r0, cols - c0], order=0,
                           mode='nearest').astype(np.uint8)


def woody_mask(classes):
    """True where tree cover or shrubland."""
    return (classes == TREE) | (classes == SHRUB)


def summarise(classes):
    """{class name: fraction} for an AOI, most common first."""
    vals, counts = np.unique(classes, return_counts=True)
    total = counts.sum()
    out = {CLASS_NAMES.get(int(v), str(int(v))): float(c) / total
           for v, c in zip(vals, counts)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
