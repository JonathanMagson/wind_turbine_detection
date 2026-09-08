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


def _tile_name_from_corner(t_lon, t_lat):
    ns = 'N' if t_lat >= 0 else 'S'
    ew = 'E' if t_lon >= 0 else 'W'
    return 'ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
        ns, abs(t_lat), ew, abs(t_lon))


def tile_name(lon, lat):
    """WorldCover tile covering a point, named by its south-west corner."""
    return _tile_name_from_corner(int(math.floor(lon / 3.0) * 3),
                                  int(math.floor(lat / 3.0) * 3))


def tiles_for_bbox(bbox):
    """Every 3x3 degree tile needed to cover ``bbox``.

    An AOI is not guaranteed to sit inside one tile -- a boundary falls every
    3 degrees, and NSW has one at latitude -30, right through the Brigalow Belt.
    """
    w, s, e, n = bbox
    lon0 = int(math.floor(w / 3.0) * 3)
    lon1 = int(math.floor((e - 1e-9) / 3.0) * 3)
    lat0 = int(math.floor(s / 3.0) * 3)
    lat1 = int(math.floor((n - 1e-9) / 3.0) * 3)
    return [(lon, lat)
            for lon in range(lon0, lon1 + 3, 3)
            for lat in range(lat0, lat1 + 3, 3)]


def read_aoi(bbox, pixel_deg):
    """Land cover class codes resampled onto the AOI grid (nearest neighbour).

    Mosaics across tile boundaries. Raises if any part of the AOI could not be
    filled, rather than returning a silently wrong mask -- reading a single
    tile and clamping out-of-range coordinates yields plausible-looking
    nonsense for the part of the AOI that falls in the neighbouring tile.
    """
    from rasterio.windows import Window

    from .reader import _prepare_gdal_env, aoi_grid
    _prepare_gdal_env()
    import rasterio

    lon_g, lat_g = aoi_grid(bbox, pixel_deg)
    out = np.zeros(lon_g.shape, dtype=np.uint8)
    filled = np.zeros(lon_g.shape, dtype=bool)

    for tlon, tlat in tiles_for_bbox(bbox):
        name = _tile_name_from_corner(tlon, tlat)
        url = f'/vsicurl/{BUCKET}/{VERSION}/map/{name}'
        try:
            ds = rasterio.open(url)
        except Exception:                                    # noqa: BLE001
            continue                       # ocean tiles are simply not published
        with ds:
            # rasterio's ds.index() is scalar-only here, so invert the north-up
            # affine directly: col = (lon - c) / a, row = (lat - f) / e.
            a, _, c, _, e_, f = ds.transform[:6]
            cols = np.floor((lon_g - c) / a).astype(int)
            rows = np.floor((lat_g - f) / e_).astype(int)
            inside = ((rows >= 0) & (rows < ds.height)
                      & (cols >= 0) & (cols < ds.width) & ~filled)
            if not inside.any():
                continue
            r0, r1 = rows[inside].min(), rows[inside].max() + 1
            c0, c1 = cols[inside].min(), cols[inside].max() + 1
            block = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            out[inside] = block[rows[inside] - r0, cols[inside] - c0]
            filled |= inside

    if not filled.all():
        raise ValueError(
            'WorldCover covers only %.1f%% of the AOI; %d tile(s) were tried. '
            'The rest is most likely ocean, which WorldCover does not publish.'
            % (100.0 * filled.mean(), len(tiles_for_bbox(bbox))))
    return out


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
