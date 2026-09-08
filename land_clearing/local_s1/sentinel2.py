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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import urllib.parse
import urllib.request

import numpy as np

from . import http_util

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
        data = json.loads(http_util.fetch_text(url, timeout=90))
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


@lru_cache(maxsize=1024)
def _granule_files(safe_prefix):
    """Every IMG_DATA file in a .SAFE, listed once and cached.

    Listing per band instead costs six API round trips per scene before a
    single pixel is read, which dominated the runtime of the fusion stage.
    """
    granules = _ls(safe_prefix + 'GRANULE/')
    if not granules:
        return ()
    return tuple(_ls(granules[0] + 'IMG_DATA/', delimiter=''))


def _band_url(safe_prefix, resolution, suffix):
    for f in _granule_files(safe_prefix):
        if f.endswith(suffix) and ('/%s/' % resolution) in f:
            return RAW + f
    return None


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


def find_least_cloudy(bbox, candidates, limit=24, min_valid=0.6,
                      verbose=True):
    """Best available granule when nothing is genuinely clear.

    ``find_cloudfree`` returns nothing when no scene meets a hard cloud bar,
    which is the right answer for a measurement and the wrong one for a
    picture: a 20% cloudy chip still shows whether a paddock was cleared. The
    cloud fraction comes back with it so it can be stated rather than implied.
    """
    best = None
    tried = 0
    for date, prefix in candidates:
        if tried >= limit:
            break
        res = assess(prefix, bbox)
        if res is None:
            continue
        valid, cloud = res
        tried += 1
        if valid < min_valid:
            continue
        if best is None or cloud < best[2]:
            best = (date, prefix, cloud)
        if cloud <= 0.01:
            break
    if best and verbose:
        print('  fallback %s at %.0f%% cloud' % (best[0], 100 * best[2]))
    return best if best else (None, None, None)


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


def read_ndvi(safe_prefix, bbox):
    """Cloud-masked NDVI over the AOI, with the window's transform and CRS.

    Returns ``None`` if the granule has no pixels here. Cloudy and shadowed
    pixels come back as NaN so a stack of dates can be median-composited
    without them.
    """
    import rasterio
    from rasterio.windows import transform as window_transform

    red_url = _band_url(safe_prefix, 'R10m', 'B04_10m.jp2')
    nir_url = _band_url(safe_prefix, 'R10m', 'B08_10m.jp2')
    scl_url = _band_url(safe_prefix, 'R20m', 'SCL_20m.jp2')
    if not (red_url and nir_url and scl_url):
        return None

    with rasterio.open(red_url) as ds:
        win = _window(ds, bbox)
        red = ds.read(1, window=win).astype(np.float32)
        tfm = window_transform(win, ds.transform)
        crs = ds.crs
    if red.size == 0:
        return None
    with rasterio.open(nir_url) as ds:
        nir = ds.read(1, window=_window(ds, bbox)).astype(np.float32)
    with rasterio.open(scl_url) as ds:
        scl = ds.read(1, window=_window(ds, bbox))

    # SCL is 20 m; repeat it to the 10 m grid and trim to match.
    scl10 = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
    h = min(red.shape[0], nir.shape[0], scl10.shape[0])
    w = min(red.shape[1], nir.shape[1], scl10.shape[1])
    red, nir, scl10 = red[:h, :w], nir[:h, :w], scl10[:h, :w]

    bad = np.isin(scl10, SCL_CLOUD) | (scl10 == SCL_NODATA)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
    ndvi[bad] = np.nan
    ndvi[~np.isfinite(ndvi)] = np.nan
    return {'ndvi': ndvi, 'transform': tfm, 'crs': crs}


def ndvi_composite(bbox, granules, max_scenes=3, min_valid=0.35, workers=8):
    """Median cloud-free NDVI over up to ``max_scenes`` granules.

    Compositing rather than taking a single date matters here: a single clear
    scene still carries that day's soil moisture and sun angle, and the whole
    point of the comparison is to isolate a persistent change from those.

    Candidates are read in parallel because most of the cost is network
    latency, and roughly half of them turn out to be empty or cloudy over any
    given chip -- so more are fetched than are needed.
    """
    # Cloudy windows need a deep candidate list: over Cobar in early July
    # several consecutive dates came back 0%% valid.
    candidates = list(granules)[:max_scenes * 6]
    if not candidates:
        return None

    def one(item):
        date, prefix = item
        try:
            got = read_ndvi(prefix, bbox)
        except Exception:                                    # noqa: BLE001
            return None
        if got is None or np.isfinite(got['ndvi']).mean() < min_valid:
            return None
        got['date'] = date
        return got

    with ThreadPoolExecutor(workers) as ex:
        results = [r for r in ex.map(one, candidates) if r]
    if not results:
        return None
    results = results[:max_scenes]

    h = min(r['ndvi'].shape[0] for r in results)
    w = min(r['ndvi'].shape[1] for r in results)
    stack = np.stack([r['ndvi'][:h, :w] for r in results])
    with np.errstate(invalid='ignore'):
        comp = np.nanmedian(stack, axis=0)
    return {'ndvi': comp, 'transform': results[0]['transform'],
            'crs': results[0]['crs'], 'dates': [r['date'] for r in results]}
