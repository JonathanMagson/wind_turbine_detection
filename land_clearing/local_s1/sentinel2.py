"""Cloud-free Sentinel-2 L2A chips from the public Google Cloud mirror.

Used to show what the radar detections look like in optical imagery. The
Sentinel-2 archive on GCS lists and serves anonymously, like the Sentinel-1
mirror, so this needs no credentials either.

Two traps in the archive layout:

  * A date can have several .SAFE granules -- reprocessings under different
    baselines, and partial ones. Some contain no pixels at all over a given
    AOI, so a granule has to be probed rather than assumed.
  * Granules are per MGRS tile in UTM, not lat/lon, so the AOI has to be
    reprojected before a window can be cut.

Cloud screening uses the L2A scene classification band (SCL) over the AOI
itself, not the scene-level metadata percentage: a granule can be 40% cloudy
overall and perfectly clear over a 3 km chip.
"""

import json
import re
import urllib.parse
import urllib.request

import numpy as np

BUCKET = 'gcp-public-data-sentinel-2'
API = 'https://storage.googleapis.com/storage/v1/b/%s/o' % BUCKET
RAW = '/vsicurl/https://storage.googleapis.com/%s/' % BUCKET

# SCL classes that are not usable ground: shadow, cloud medium/high, cirrus.
SCL_CLOUD = (3, 8, 9, 10)
SCL_NODATA = 0


def _ls(prefix, delimiter='/'):
    out, token = [], None
    while True:
        url = ('%s?prefix=%s&delimiter=%s&maxResults=1000'
               % (API, urllib.parse.quote(prefix), delimiter))
        if token:
            url += '&pageToken=' + token
        data = json.load(urllib.request.urlopen(url, timeout=90))
        out += data.get('prefixes', []) + [i['name'] for i in data.get('items', [])]
        token = data.get('nextPageToken')
        if not token:
            return out


def mgrs_tile(lon, lat):
    """MGRS 100 km square for a point, as (zone, band, square)."""
    from pyproj import Transformer
    zone = int((lon + 180) // 6) + 1
    band = 'CDEFGHJKLMNPQRSTUVWX'[int((lat + 80) // 8)]
    epsg = (32700 if lat < 0 else 32600) + zone
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:%d' % epsg,
                                always_xy=True).transform(lon, lat)
    col = 'ABCDEFGH'[int(x // 100000) - 1]
    row = 'ABCDEFGHJKLMNPQRSTUV'[int(y // 100000) % 20]
    return zone, band, col + row


def list_granules(zone, band, square, start=None, end=None):
    """L2A .SAFE prefixes for a tile, as (date, prefix), sorted by date."""
    prefix = 'L2/tiles/%d/%s/%s/' % (zone, band, square)
    out = []
    for p in _ls(prefix):
        m = re.search(r'_MSIL2A_(\d{8})T', p)
        if not m:
            continue
        date = m.group(1)
        if start and date < start.replace('-', ''):
            continue
        if end and date > end.replace('-', ''):
            continue
        out.append((date, p))
    return sorted(out)


def _band_url(safe_prefix, resolution, suffix):
    granules = _ls(safe_prefix + 'GRANULE/')
    if not granules:
        return None
    files = _ls(granules[0] + 'IMG_DATA/%s/' % resolution, delimiter='')
    hits = [f for f in files if f.endswith(suffix)]
    return RAW + hits[0] if hits else None


def _window(ds, bbox):
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    tr = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
    w, s, e, n = bbox
    xs, ys = tr.transform([w, e, e, w], [n, n, s, s])
    return from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform)


def assess(safe_prefix, bbox):
    """Return (valid_fraction, cloud_fraction) over the AOI, or None if absent."""
    import rasterio
    url = _band_url(safe_prefix, 'R20m', 'SCL_20m.jp2')
    if not url:
        return None
    with rasterio.open(url) as ds:
        scl = ds.read(1, window=_window(ds, bbox))
    if scl.size == 0:
        return None
    valid = scl != SCL_NODATA
    if not valid.any():
        return 0.0, 1.0
    cloud = np.isin(scl, SCL_CLOUD)
    return float(valid.mean()), float(cloud[valid].mean())


def read_rgb(safe_prefix, bbox, with_nir=True):
    """True-colour (and optionally NIR) reflectance over the AOI.

    Returns a dict of 2-D float arrays scaled to reflectance, plus the window
    bounds in lon/lat so the chip can be plotted against vector overlays.
    """
    import rasterio
    from pyproj import Transformer

    wanted = {'red': 'B04_10m.jp2', 'green': 'B03_10m.jp2', 'blue': 'B02_10m.jp2'}
    if with_nir:
        wanted['nir'] = 'B08_10m.jp2'

    out, extent = {}, None
    for key, suffix in wanted.items():
        url = _band_url(safe_prefix, 'R10m', suffix)
        if not url:
            return None
        with rasterio.open(url) as ds:
            win = _window(ds, bbox)
            out[key] = ds.read(1, window=win).astype(np.float32) / 10000.0
            if extent is None:
                b = rasterio.windows.bounds(win, ds.transform)
                tr = Transformer.from_crs(ds.crs, 'EPSG:4326', always_xy=True)
                xs, ys = tr.transform([b[0], b[2]], [b[1], b[3]])
                extent = [min(xs), max(xs), min(ys), max(ys)]
    out['extent'] = extent
    return out


def find_cloudfree(bbox, candidates, max_cloud=0.02, min_valid=0.98, limit=12,
                   verbose=True):
    """First granule from ``candidates`` that is clear over the AOI.

    ``candidates`` is a list of (date, prefix) in the order you want tried --
    so pass it reversed to prefer the latest date before an event, or forward
    to prefer the earliest after it.
    """
    tried = 0
    for date, prefix in candidates:
        if tried >= limit:
            break
        res = assess(prefix, bbox)
        if res is None:
            continue
        valid, cloud = res
        tried += 1
        if verbose:
            print('  %s valid %3.0f%% cloud %3.0f%%'
                  % (date, 100 * valid, 100 * cloud))
        if valid >= min_valid and cloud <= max_cloud:
            return date, prefix
    return None, None
