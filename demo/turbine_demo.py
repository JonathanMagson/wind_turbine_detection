#!/usr/bin/env python3
"""Build (and optionally serve) the in-browser wind turbine detection demo.

    # detect turbines in the bundled Sentinel-2 scene and open the visual
    python3 demo/turbine_demo.py --serve

    # the small IPOL test crop instead
    python3 demo/turbine_demo.py --image example/test.tif \
        --meta example/test_meta.txt --serve

With ``--serve`` the page also gets a live detector: moving the shadow or hub
threshold re-runs ``main_NFA`` on the server and streams a fresh -log10(NFA)
map back into the canvas.  Without it the page is a single self-contained HTML
file -- the NFA threshold still re-detects in the browser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import pipeline as pl  # noqa: E402

TEMPLATE = os.path.join(HERE, "template.html")

HTML_SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
{fragment}
</body>
</html>
"""


def build_payload(args) -> tuple[dict, dict]:
    """Run the detector once and pack everything the page needs."""
    scene = pl.read_scene(args.image, args.meta)
    sat_azi, sat_zen, band = pl.satellite_angles(scene, args.band)
    sun = pl.sun_angles(scene, args.sun_time)
    image, valid = pl.load_image(scene)

    t0 = time.perf_counter()
    nfa = pl.run_detector(image, sun, sat_azi, sat_zen, args.heights,
                          args.t_shadow, args.t_hub)
    elapsed = time.perf_counter() - t0

    detections = pl.cluster_detections(nfa, valid, scene, args.t_nfa)
    lat, lon = scene.center_latlon

    payload = {
        "scene": {
            "name": args.name or os.path.basename(args.image),
            "image": os.path.relpath(args.image, ROOT),
            "meta": os.path.relpath(args.meta, ROOT),
            "width": scene.width,
            "height": scene.height,
            "resolution": pl.RESOLUTION_M,
            "band": band,
            "satellite": scene.tags.get("satellite", "Sentinel-2"),
            "level": scene.tags.get("processing_level", "unknown"),
            "tile": scene.tags.get("mgrs_id", ""),
            "acquired": sun["timestamp"],
            "timeSource": sun["source"],
            "timeCandidates": sun["candidates"],
            "utmZone": f"{scene.zone_number}{'N' if scene.northern else 'S'}",
            "center": [round(lat, 5), round(lon, 5)],
            "corners": scene.corner_latlon(),
            "nodata": scene.nodata,
            "nodataPixels": int((~valid).sum()),
        },
        "geometry": {
            "sunAzimuth": round(sun["azimuth"], 3),
            "sunAltitude": round(sun["altitude"], 3),
            "satAzimuth": round(sat_azi, 3),
            "satZenith": round(sat_zen, 3),
            "heights": list(args.heights),
            "sampling": pl.sampling_geometry(sun, sat_azi, sat_zen,
                                             args.heights[0]),
        },
        "params": {
            "tNFA": args.t_nfa,
            "tShadow": args.t_shadow,
            "tHub": args.t_hub,
            "shadowNeighbourM": pl.SHADOW_NEIGH_M,
            "hubNeighbourM": pl.HUB_NEIGH_M,
            "sampleRateM": pl.SAMPLE_RATE_M,
        },
        "stats": pl.detector_statistics(image, sun, sat_azi, sat_zen,
                                        args.heights[0], args.t_shadow,
                                        args.t_hub, len(args.heights)),
        "runtime": {"seconds": round(elapsed, 2), "live": bool(args.serve)},
        "reflectance": pl.encode_reflectance(image, valid),
        "nfa": pl.encode_nfa(nfa, valid),
        "histogram": pl.nfa_histogram(nfa, valid),
        "detections": detections,
    }

    state = {"scene": scene, "image": image, "valid": valid, "nfa": nfa,
             "sun": sun, "sat": (sat_azi, sat_zen)}
    return payload, state


def page_title(name: str) -> str:
    """One page per scene, so the scene names the page."""
    name = (name or "").strip()
    if not name or name.lower().endswith(".tif"):
        return "Turbine Shadow Census"
    return f"{name.split(',')[0].strip()} Shadow Census"


def render_html(payload: dict, standalone: bool) -> str:
    with open(TEMPLATE) as fh:
        fragment = fh.read()
    data = json.dumps(payload, separators=(",", ":"))
    fragment = fragment.replace("__TITLE__", page_title(payload["scene"]["name"]))
    fragment = fragment.replace("__PAYLOAD__", data)
    if not standalone:
        return fragment
    return HTML_SKELETON.format(fragment=fragment)


def write_sidecars(payload: dict, outdir: str) -> list[str]:
    """Detections as CSV and GeoJSON, next to the HTML."""
    dets = payload["detections"]
    written = []

    csv_path = os.path.join(outdir, "detections.csv")
    with open(csv_path, "w") as fh:
        fh.write("id,row,col,logNFA,pixels,lat,lon,easting,northing\n")
        for d in dets:
            fh.write(f"{d['id']},{d['row']},{d['col']},{d['nfa']},{d['pixels']},"
                     f"{d['lat']},{d['lon']},{d['easting']},{d['northing']}\n")
    written.append(csv_path)

    gj_path = os.path.join(outdir, "detections.geojson")
    geo = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
            "properties": {k: v for k, v in d.items() if k not in ("lat", "lon")},
        } for d in dets],
    }
    with open(gj_path, "w") as fh:
        json.dump(geo, fh, indent=1)
    written.append(gj_path)
    return written


class DemoHandler(BaseHTTPRequestHandler):
    """Serves the page and re-runs the detector on demand."""

    server_version = "TurbineDemo/1.0"

    def __init__(self, *a, html: str = "", state: dict | None = None,
                 args=None, **kw):
        self.html = html
        self.state = state or {}
        self.args = args
        super().__init__(*a, **kw)

    def log_message(self, fmt, *a):  # quieter console
        sys.stderr.write("  %s\n" % (fmt % a))

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(self.html.encode("utf-8"), "text/html; charset=utf-8")
        if url.path == "/api/ping":
            return self._send(json.dumps({"live": True}).encode(), "application/json")
        if url.path == "/api/detect":
            return self._detect(parse_qs(url.query))
        self.send_error(404)

    def _detect(self, q: dict):
        def num(key, default, cast=float):
            try:
                return cast(q.get(key, [default])[0])
            except (TypeError, ValueError):
                return default

        diff_s = num("diff_s", self.args.t_shadow, int)
        diff_h = num("diff_h", self.args.t_hub, int)
        t_nfa = num("t_nfa", self.args.t_nfa)
        heights = [float(h) for h in q.get("heights", [""])[0].split(",") if h] \
            or list(self.args.heights)

        st = self.state
        t0 = time.perf_counter()
        nfa = pl.run_detector(st["image"], st["sun"], st["sat"][0], st["sat"][1],
                              heights, diff_s, diff_h)
        elapsed = time.perf_counter() - t0
        body = json.dumps({
            "nfa": pl.encode_nfa(nfa, st["valid"], fast=True),
            "histogram": pl.nfa_histogram(nfa, st["valid"]),
            "detections": pl.cluster_detections(nfa, st["valid"], st["scene"], t_nfa),
            "geometry": pl.sampling_geometry(st["sun"], st["sat"][0], st["sat"][1],
                                             heights[0]),
            "stats": pl.detector_statistics(st["image"], st["sun"], st["sat"][0],
                                            st["sat"][1], heights[0], diff_s,
                                            diff_h, len(heights)),
            "params": {"tShadow": diff_s, "tHub": diff_h, "tNFA": t_nfa,
                       "heights": heights},
            "runtime": {"seconds": round(elapsed, 2)},
        }, separators=(",", ":")).encode()
        self._send(body, "application/json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="In-browser demo of a-contrario wind turbine detection.")
    p.add_argument("--image", default=os.path.join(ROOT, "band2_masked.tif"),
                   help="Sentinel-2 single-band GeoTIFF")
    p.add_argument("--meta", default=os.path.join(ROOT, "band2_metadata.txt"),
                   help="gdalinfo-style metadata sidecar")
    p.add_argument("--name", default=None, help="label shown in the page header")
    p.add_argument("--band", default="B02",
                   help="band whose viewing angles to use (default B02)")
    p.add_argument("--heights", default="80",
                   help="comma-separated hub heights to test, in metres")
    p.add_argument("--t-nfa", type=float, default=1.0,
                   help="-log10(NFA) detection threshold (default 1)")
    p.add_argument("--t-shadow", type=int, default=25, help="shadow threshold")
    p.add_argument("--t-hub", type=int, default=50, help="hub threshold")
    p.add_argument("--sun-time", default="auto",
                   choices=["auto", "granule_date", "date"],
                   help="which metadata timestamp fixes the sun position")
    p.add_argument("--out", default=os.path.join(HERE, "out"),
                   help="output directory for index.html and sidecars")
    p.add_argument("--fragment", action="store_true",
                   help="also emit a head-less HTML fragment for embedding")
    p.add_argument("--serve", action="store_true",
                   help="serve the page with a live detector endpoint")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true",
                   help="with --serve, do not launch a browser")
    args = p.parse_args(argv)
    args.heights = [float(h) for h in str(args.heights).split(",") if h.strip()]

    print(f"scene    {args.image}")
    payload, state = build_payload(args)
    sc, geo = payload["scene"], payload["geometry"]
    print(f"         {sc['width']}x{sc['height']} px  {sc['satellite']} {sc['band']}"
          f"  tile {sc['tile']}  {sc['acquired']} ({sc['timeSource']})")
    print(f"sun      azimuth {geo['sunAzimuth']:.1f} deg  altitude "
          f"{geo['sunAltitude']:.1f} deg -> shadow "
          f"{geo['sampling']['shadowLengthM']:.0f} m for a "
          f"{geo['heights'][0]:.0f} m hub")
    print(f"detector {geo['sampling']['samples']} samples per pixel, "
          f"{payload['runtime']['seconds']}s")
    print(f"found    {len(payload['detections'])} candidates at "
          f"-log10(NFA) > {args.t_nfa}")

    os.makedirs(args.out, exist_ok=True)
    html = render_html(payload, standalone=True)
    index = os.path.join(args.out, "index.html")
    with open(index, "w") as fh:
        fh.write(html)
    outputs = [index] + write_sidecars(payload, args.out)
    if args.fragment:
        frag = os.path.join(args.out, "fragment.html")
        with open(frag, "w") as fh:
            fh.write(render_html(payload, standalone=False))
        outputs.append(frag)
    for path in outputs:
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(".."):
            rel = path
        size = os.path.getsize(path)
        print(f"wrote    {rel}" + (f" ({size / 1e6:.1f} MB)" if size > 1e6 else ""))

    if not args.serve:
        return 0

    payload["runtime"]["live"] = True
    live_html = render_html(payload, standalone=True)
    handler = partial(DemoHandler, html=live_html, state=state, args=args)
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    url = f"http://localhost:{args.port}/"
    print(f"\nserving  {url}  (Ctrl-C to stop)")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
