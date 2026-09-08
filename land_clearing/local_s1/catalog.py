"""Discover Sentinel-1 GRD scenes in the public AWS archive.

The ESA Sentinel-1 Level-1 archive is mirrored to the ``sentinel-s1-l1c`` S3
bucket, which lists and serves anonymously -- no credentials, no requester-pays.
This module finds the scenes covering an AOI on a single relative orbit, which
is what the omnibus test needs (constant local incidence angle per pixel).

Keys are laid out as ``GRD/<year>/<month>/<day>/<mode>/<pol>/<scene>/`` with
month and day *not* zero padded. Each scene carries a small ``productInfo.json``
with its footprint and absolute orbit number.

Scene discovery is cheap because Sentinel-1 repeats every 12 days: find one
covering scene, then step 12 days at a time rather than searching every date.
"""

import datetime
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import http_util

BUCKET = 'https://sentinel-s1-l1c.s3.amazonaws.com'
REPEAT_DAYS = 12


def _get(url, timeout=90):
    return http_util.fetch_text(url, timeout=timeout)


def relative_orbit(absolute_orbit, mission_id):
    """Relative orbit (track) number from the absolute orbit."""
    if mission_id == 'S1A':
        return ((absolute_orbit - 73) % 175) + 1
    return ((absolute_orbit - 27) % 175) + 1


def list_scene_prefixes(date, mode='IW', pol='DV'):
    """All scene prefixes archived for a date."""
    prefix = f'GRD/{date.year}/{date.month}/{date.day}/{mode}/{pol}/'
    out, token = [], None
    while True:
        url = f'{BUCKET}/?list-type=2&prefix={prefix}&delimiter=/&max-keys=1000'
        if token:
            url += '&continuation-token=' + urllib.parse.quote(token, safe='')
        body = _get(url)
        out += re.findall(
            r'<CommonPrefixes><Prefix>(.*?)</Prefix></CommonPrefixes>', body)
        nxt = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>',
                        body)
        if not nxt:
            break
        token = nxt.group(1)
    return out


def product_info(prefix):
    """``productInfo.json`` for a scene prefix, or None if unavailable."""
    try:
        return json.loads(_get(f'{BUCKET}/{prefix}productInfo.json'))
    except Exception:                                        # noqa: BLE001
        return None


def scene_time(prefix):
    """UTC time-of-day as an integer HHMMSS, parsed from the scene name."""
    return int(re.search(r'_\d{8}T(\d{6})_', prefix).group(1))


def find_covering_scenes(aoi, date, utc_window=None, mode='IW', pol='DV',
                         orbit=None, workers=16):
    """Scenes on ``date`` whose footprint fully contains ``aoi``.

    ``aoi`` is a shapely geometry in EPSG:4326. ``utc_window`` is an optional
    ``(hhmmss_min, hhmmss_max)`` prefilter -- a huge saving, since it avoids
    fetching footprints for the whole global archive of that day.
    """
    from shapely.geometry import shape

    prefixes = list_scene_prefixes(date, mode, pol)
    if utc_window:
        lo, hi = utc_window
        prefixes = [p for p in prefixes if lo <= scene_time(p) <= hi]
    if not prefixes:
        return []

    with ThreadPoolExecutor(workers) as ex:
        infos = [i for i in ex.map(product_info, prefixes) if i]

    out = []
    for i in infos:
        if not shape(i['footprint']).contains(aoi):
            continue
        i['relativeOrbit'] = relative_orbit(i['absoluteOrbitNumber'],
                                            i['missionId'])
        if orbit is not None and i['relativeOrbit'] != orbit:
            continue
        out.append(i)
    return out


def find_series(aoi, start, end, orbit=None, utc_window=None, seed_search_days=13,
                mode='IW', pol='DV', workers=16, verbose=True):
    """Find a single-track series covering ``aoi`` between ``start`` and ``end``.

    Scans forward day by day from ``start`` until it finds a covering scene
    (at most ``seed_search_days``, slightly more than one repeat cycle), then
    steps by the 12-day repeat, which makes the rest of the search nearly free.

    Returns a list of productInfo dicts sorted by acquisition time.
    """
    start = _as_date(start)
    end = _as_date(end)

    seed = None
    for offset in range(seed_search_days):
        day = start + datetime.timedelta(days=offset)
        if day >= end:
            break
        hits = find_covering_scenes(aoi, day, utc_window, mode, pol, orbit,
                                    workers)
        if hits:
            seed = hits[0]
            if verbose:
                print('seed scene %s (relative orbit %d) on %s'
                      % (seed['id'], seed['relativeOrbit'], day))
            break
    if seed is None:
        raise ValueError(
            'No scene fully containing the AOI in the %d days from %s. The AOI '
            'may straddle two slices (try a smaller one), or this track may '
            'not cover it.' % (seed_search_days, start))

    track = seed['relativeOrbit']
    # Keep the search window tight around the seed's time of day; the same
    # track always crosses at nearly the same UTC time.
    t = scene_time(seed['path'] + '/')
    window = (max(0, t - 3000), min(235959, t + 3000))

    series, day = [], _as_date(seed['startTime'][:10])
    while day < end:
        hits = find_covering_scenes(aoi, day, window, mode, pol, track, workers)
        if hits:
            series.append(hits[0])
            if verbose:
                print('  %s  %s' % (hits[0]['startTime'], hits[0]['id'][:44]))
        day += datetime.timedelta(days=REPEAT_DAYS)

    series.sort(key=lambda i: i['startTime'])
    return series


def _as_date(d):
    if isinstance(d, datetime.date):
        return d
    return datetime.date.fromisoformat(str(d)[:10])
