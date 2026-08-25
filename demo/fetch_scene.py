#!/usr/bin/env python3
"""Cut a Sentinel-2 scene out of the public AWS COG archive, ready to detect.

    python3 demo/fetch_scene.py --lat -35.13 --lon 149.53 --km 14 \
        --year 2024 --months 5,6,7,8 --name capital

Reads only the window it needs -- the source COGs are 10980 x 10980 -- and
writes a GeoTIFF plus a `gdalinfo`-style sidecar in the format the detector's
loader already understands, including the per-band viewing angles from the
granule metadata.

Winter months are worth asking for in the southern hemisphere: a low sun throws
a longer shadow, which is the whole signal this detector runs on.

Needs `pip install -r demo/requirements-fetch.txt` on top of the demo's own
requirements. No credentials: the bucket is public.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

import mgrs
import numpy as np
import rasterio
import utm
from rasterio.windows import Window

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
PREFIX = "sentinel-s2-l2a-cogs"
RESOLUTION_M = 10.0

# Sentinel-2 band ids as they appear in granule_metadata.xml, in order.
BAND_IDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]

# Scene classification classes that make a window useless to this detector:
# cloud shadow, cloud (medium and high probability), thin cirrus.
SCL_CLOUDY = (3, 8, 9, 10)


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return fh.read()


def list_scenes(tile: str, year: int, months: list[int]) -> list[str]:
    """Scene ids under one MGRS tile, e.g. S2B_55HGB_20240612_0_L2A."""
    zone, band, square = tile[:-3], tile[-3], tile[-2:]
    found: list[str] = []
    for month in months:
        prefix = f"{PREFIX}/{zone}/{band}/{square}/{year}/{month}/"
        url = f"{BUCKET}/?list-type=2&delimiter=/&prefix={prefix}&max-keys=200"
        body = _get(url).decode("utf-8", "replace")
        for m in re.finditer(r"<Prefix>([^<]+)</Prefix>", body):
            leaf = m.group(1).rstrip("/").rsplit("/", 1)[-1]
            if leaf.endswith("_L2A"):
                found.append(f"{prefix}{leaf}")
    return sorted(found)


SQUARE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"


def neighbour_tiles(tile: str) -> list[str]:
    """The tile itself, then the tiles around it, nearest first.

    The Sentinel-2 grid names its tiles after MGRS squares, but a point within
    a few kilometres of a latitude-band edge converts to a square the archive
    files under the *neighbouring* band -- Silverton, at 31.93S, converts to
    54JWK while the imagery lives under 54HWK. So the caller walks this list
    and keeps the first tile whose raster actually contains the point.
    """
    zone, band, east, north = tile[:-3], tile[-3], tile[-2], tile[-1]
    ib = BAND_LETTERS.index(band)
    ie, jn = SQUARE_LETTERS.index(east), SQUARE_LETTERS.index(north)
    out = []
    for db in (0, -1, 1):
        for de, dn in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            b, e, n = ib + db, ie + de, jn + dn
            if (0 <= b < len(BAND_LETTERS) and 0 <= e < len(SQUARE_LETTERS)
                    and 0 <= n < len(SQUARE_LETTERS)):
                out.append(f"{zone}{BAND_LETTERS[b]}"
                           f"{SQUARE_LETTERS[e]}{SQUARE_LETTERS[n]}")
    return out


def covers(path: str, band: str, lat: float, lon: float) -> bool:
    """Does this scene's raster actually contain the point?"""
    east, north, _, _ = utm.from_latlon(lat, lon)
    try:
        with rasterio.open(f"{BUCKET}/{path}/{band}.tif") as src:
            b = src.bounds
    except Exception:
        return False
    return b.left <= east <= b.right and b.bottom <= north <= b.top


def scene_metadata(path: str) -> dict:
    """The STAC item that sits beside the rasters."""
    scene = path.rstrip("/").rsplit("/", 1)[-1]
    item = json.loads(_get(f"{BUCKET}/{path}/{scene}.json"))
    p = item["properties"]
    return {
        "path": path,
        "scene": scene,
        "datetime": p["datetime"],
        "cloud": float(p.get("eo:cloud_cover", 100.0)),
        "platform": p.get("platform", "sentinel-2"),
        "epsg": int(p.get("proj:epsg", 0)),
        "tile": p.get("grid:code", "").replace("MGRS-", ""),
        "sun_azimuth": float(p.get("view:sun_azimuth", float("nan"))),
        "sun_elevation": float(p.get("view:sun_elevation", float("nan"))),
        "nodata_pct": float(p.get("s2:nodata_pixel_percentage", 0.0)),
    }


def viewing_angles(path: str) -> tuple[dict, dict, str]:
    """Per-band satellite azimuth and zenith, straight from the granule XML."""
    root = ET.fromstring(_get(f"{BUCKET}/{path}/granule_metadata.xml"))
    azimuth: dict[str, float] = {}
    zenith: dict[str, float] = {}
    sensing = ""
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "SENSING_TIME":
            sensing = (el.text or "").strip()
        elif tag == "Mean_Viewing_Incidence_Angle":
            idx = int(el.attrib.get("bandId", -1))
            if not 0 <= idx < len(BAND_IDS):
                continue
            vals = {c.tag.split("}")[-1]: float(c.text) for c in el}
            azimuth[BAND_IDS[idx]] = vals["AZIMUTH_ANGLE"]
            zenith[BAND_IDS[idx]] = vals["ZENITH_ANGLE"]
    return azimuth, zenith, sensing


def window_quality(path: str, lat: float, lon: float, km: float) -> tuple[float, float]:
    """Cloud and no-data fractions inside the window, from the 20 m SCL band.

    Tile-level statistics describe 12000 square kilometres: a 14 km window can
    be solid cloud inside a tile reporting 0.3%, and a tile that is 18% empty
    because the swath clipped its corner can still cover the window completely.
    Both questions are better asked of the footprint we actually want.
    """
    data, *_ = cut_window(path, "SCL", lat, lon, km)
    return (float(np.isin(data, SCL_CLOUDY).mean()), float((data == 0).mean()))


def cut_window(path: str, band: str, lat: float, lon: float, km: float):
    """Read a km-wide square centred on (lat, lon) out of the band's COG."""
    url = f"{BUCKET}/{path}/{band}.tif"
    with rasterio.open(url) as src:
        east, north, _, _ = utm.from_latlon(lat, lon)
        row, col = src.index(east, north)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(
                f"({lat}, {lon}) falls outside {path.rsplit('/', 1)[-1]}; "
                "the window would be silently clipped to an edge")
        side = int(round(km * 1000 / RESOLUTION_M))
        col0 = int(np.clip(col - side // 2, 0, src.width - side))
        row0 = int(np.clip(row - side // 2, 0, src.height - side))
        window = Window(col0, row0, min(side, src.width), min(side, src.height))
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        profile = dict(src.profile)
        nodata = src.nodata
    profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                   count=1, dtype=data.dtype, transform=transform,
                   compress="deflate", tiled=True)
    return data, transform, profile, nodata


def write_sidecar(path_txt: str, tif_name: str, data, transform, epsg: int,
                  meta: dict, azimuth: dict, zenith: dict, sensing: str,
                  nodata) -> None:
    """Emit the `gdalinfo` text layout the detector's loader parses."""
    height, width = data.shape
    zone = epsg % 100
    northern = 32600 < epsg < 32700
    hemi = "N" if northern else "S"
    ox, oy = transform.c, transform.f

    def corner(row, col):
        e = ox + col * RESOLUTION_M
        n = oy - row * RESOLUTION_M
        lat, lon = utm.to_latlon(e, n, zone, northern=northern, strict=False)
        return e, n, lat, lon

    def dms(v, pos, neg):
        hemi_ = pos if v >= 0 else neg
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        s = (v - d - m / 60) * 3600
        return f"{d:3d}d{m:2d}'{s:5.2f}\"{hemi_}"

    corners = {
        "Upper Left": corner(0, 0),
        "Lower Left": corner(height, 0),
        "Upper Right": corner(0, width),
        "Lower Right": corner(height, width),
        "Center": corner(height / 2, width / 2),
    }
    stamp = sensing or meta["datetime"]
    granule = stamp.replace("T", " ").rstrip("Z")[:19]

    lines = [
        "Driver: GTiff/GeoTIFF",
        f"Files: {tif_name}",
        f"Size is {width}, {height}",
        "Coordinate System is:",
        f'PROJCRS["WGS 84 / UTM zone {zone}{hemi}",',
        '    BASEGEOGCRS["WGS 84",',
        '        DATUM["World Geodetic System 1984",',
        '            ELLIPSOID["WGS 84",6378137,298.257223563,',
        '                LENGTHUNIT["metre",1]]],',
        '        ID["EPSG",4326]],',
        f'    ID["EPSG",{epsg}]]',
        "Data axis to CRS axis mapping: 1,2",
        f"Origin = ({ox:.15f},{oy:.15f})",
        f"Pixel Size = ({RESOLUTION_M:.15f},{-RESOLUTION_M:.15f})",
        "Metadata:",
        "  AREA_OR_POINT=Area",
        f"  date={meta['datetime']}",
        f"  downloaded_by=demo/fetch_scene.py on "
        f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        f"  epsg={epsg}",
        f"  granule_date={granule}",
        f"  mgrs_id={meta['tile']}",
        "  processing_level=Level-2A",
        f"  satellite={meta['platform'].replace('sentinel-2', 'S2').upper()}",
        f"  satellite_azimuth={azimuth}",
        f"  satellite_zenith={zenith}",
        f"  source={BUCKET}/{meta['path']}",
        f"  title={meta['scene']}",
        f"  utm_zone={zone}",
        "Image Structure Metadata:",
        "  INTERLEAVE=BAND",
        "Corner Coordinates:",
    ]
    for name, (e, n, lat, lon) in corners.items():
        lines.append(f"{name:<12}( {e:12.3f}, {n:12.3f}) "
                     f"({dms(lon, 'E', 'W')}, {dms(lat, 'N', 'S')})")
    lines.append("Band 1 Block=1024x1024 Type=UInt16, ColorInterp=Gray")
    if nodata is not None:
        lines.append(f"  NoData Value={int(nodata)}")
    with open(path_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Cut a Sentinel-2 window out of the public AWS COG archive.")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--km", type=float, default=14.0, help="window side in km")
    p.add_argument("--band", default="B02", help="band to download (default B02)")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--months", default="5,6,7,8",
                   help="months to consider; winter means long shadows")
    p.add_argument("--max-cloud", type=float, default=60.0,
                   help="reject scenes whose whole tile is cloudier than this")
    p.add_argument("--max-window-cloud", type=float, default=2.0,
                   help="reject scenes whose window is cloudier than this, "
                        "measured on the 20 m classification band")
    p.add_argument("--tile", default=None, help="override the MGRS tile")
    p.add_argument("--scene", default=None, help="use this scene id exactly")
    p.add_argument("--name", required=True, help="output basename")
    p.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "scenes"))
    args = p.parse_args(argv)

    months = [int(m) for m in args.months.split(",") if m.strip()]
    if args.tile:
        wanted = [args.tile]
    else:
        wanted = neighbour_tiles(
            mgrs.MGRS().toMGRS(args.lat, args.lon, MGRSPrecision=0)[:5])

    tile, paths = None, []
    for candidate in wanted:
        found = list_scenes(candidate, args.year, months)
        if found and (args.tile or covers(found[0], args.band, args.lat, args.lon)):
            tile, paths = candidate, found
            break
        print(f"  {candidate}: {'no scenes archived' if not found else 'does not cover the point'}")
    if not paths:
        print("no archived tile covers that point in the given period",
              file=sys.stderr)
        return 1
    print(f"tile     {tile}  ({args.lat}, {args.lon})  {len(paths)} scenes")

    if args.scene:
        paths = [q for q in paths if q.endswith(args.scene)]
        if not paths:
            print(f"scene {args.scene} not in that tile/period", file=sys.stderr)
            return 1

    candidates = []
    for path in paths:
        try:
            meta = scene_metadata(path)
        except Exception as err:                      # a missing item is not fatal
            print(f"  skip {path.rsplit('/', 1)[-1]}: {err}")
            continue
        # A tile at the edge of the swath can be 99% empty and 0% cloudy, so
        # neither number alone is a filter -- but a mostly-empty tile is still
        # unlikely to cover the target, and it is cheap to drop here.
        why = ("cloud" if meta["cloud"] > args.max_cloud else
               "sliver" if meta["nodata_pct"] >= 90 else "")
        print(f"  {meta['scene']:28s} cloud {meta['cloud']:5.1f}%  "
              f"empty {meta['nodata_pct']:5.1f}%  sun {meta['sun_elevation']:4.1f}°"
              f"{'  skipped: ' + why if why else ''}")
        keep = not why
        if keep:
            candidates.append(meta)
    if not candidates:
        print("nothing under the cloud limit; raise --max-cloud", file=sys.stderr)
        return 1

    # Lowest sun first among the clear scenes: a 26 degree sun throws twice the
    # shadow a 45 degree one does, and the shadow is what gets detected.
    candidates.sort(key=lambda m: (m["sun_elevation"], m["cloud"]))
    best = None
    for meta in candidates:
        cloud, empty = window_quality(meta["path"], args.lat, args.lon, args.km)
        ok = cloud <= args.max_window_cloud / 100 and empty <= 0.01
        print(f"  window check {meta['scene']:28s} {cloud:5.1%} cloudy, "
              f"{empty:5.1%} empty{'' if ok else '  rejected'}")
        if ok:
            best = meta
            best["window_cloud"] = cloud
            break
    if best is None:
        print("every candidate is clouded over the target; widen --months or "
              "raise --max-window-cloud", file=sys.stderr)
        return 1
    print(f"chose    {best['scene']}  tile cloud {best['cloud']:.1f}%  "
          f"window cloud {best['window_cloud']:.1%}  "
          f"sun {best['sun_elevation']:.1f}° at {best['datetime']}")

    data, transform, profile, nodata = cut_window(
        best["path"], args.band, args.lat, args.lon, args.km)
    azimuth, zenith, sensing = viewing_angles(best["path"])

    os.makedirs(args.out, exist_ok=True)
    tif = os.path.join(args.out, f"{args.name}.tif")
    txt = os.path.join(args.out, f"{args.name}_meta.txt")
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data, 1)
    write_sidecar(txt, os.path.basename(tif), data, transform,
                  int(profile["crs"].to_epsg()), best, azimuth, zenith,
                  sensing, nodata)

    valid = data != (nodata if nodata is not None else 0)
    print(f"wrote    {tif}  {data.shape[1]}x{data.shape[0]} px, "
          f"{valid.sum() / data.size:.1%} valid, "
          f"DN {int(data[valid].min())}-{int(data[valid].max())}")
    print(f"wrote    {txt}")
    print(f"\nnext     python3 demo/turbine_demo.py --image {tif} "
          f"--meta {txt} --name '{args.name}' --serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
