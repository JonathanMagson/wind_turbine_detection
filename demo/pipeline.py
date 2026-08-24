#!/usr/bin/env python3
"""Scene loading, geometry and a-contrario detection for the browser demo.

Everything here is a thin wrapper around the peer-reviewed code that already
lives in the repository root (``utils_eol_compute.main_NFA``).  The demo never
reimplements the detector -- it only feeds it, clusters its output and packs
the result into something a browser can draw.
"""

from __future__ import annotations

import ast
import base64
import datetime
import io
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import tifffile
import utm
from PIL import Image
from pysolar.solar import get_altitude, get_azimuth
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils_eol_compute as uec  # noqa: E402  (repo root, added to path above)

# The detector's own defaults, kept in one place so the CLI, the HTML and the
# geometry overlay all quote the same numbers.
RESOLUTION_M = 10.0      # Sentinel-2 B02 ground sampling distance
SAMPLE_RATE_M = 10.0     # spacing between samples along the shadow
SHADOW_NEIGH_M = 15.0    # sample-to-neighbour distance across the shadow
HUB_NEIGH_M = 30.0       # hub-to-neighbour distance
N_HUB_NEIGHBOURS = 6


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

@dataclass
class Scene:
    """One Sentinel-2 crop plus everything needed to interpret it."""

    image_path: str
    meta_path: str
    width: int
    height: int
    origin: tuple[float, float]
    pixel_size: tuple[float, float]
    zone_number: int
    northern: bool
    nodata: float | None
    tags: dict[str, str] = field(default_factory=dict)

    # --- geo helpers ------------------------------------------------------
    def pixel_to_utm(self, row: float, col: float) -> tuple[float, float]:
        """Centre of pixel (row, col) in projected metres."""
        east = self.origin[0] + (col + 0.5) * self.pixel_size[0]
        north = self.origin[1] + (row + 0.5) * self.pixel_size[1]
        return east, north

    def pixel_to_latlon(self, row: float, col: float) -> tuple[float, float]:
        east, north = self.pixel_to_utm(row, col)
        lat, lon = utm.to_latlon(east, north, self.zone_number,
                                 northern=self.northern, strict=False)
        return lat, lon

    @property
    def center_latlon(self) -> tuple[float, float]:
        return self.pixel_to_latlon(self.height / 2 - 0.5, self.width / 2 - 0.5)

    def corner_latlon(self) -> dict[str, list[float]]:
        h, w = self.height, self.width
        return {
            "ul": list(self.pixel_to_latlon(-0.5, -0.5)),
            "ur": list(self.pixel_to_latlon(-0.5, w - 0.5)),
            "ll": list(self.pixel_to_latlon(h - 0.5, -0.5)),
            "lr": list(self.pixel_to_latlon(h - 0.5, w - 0.5)),
        }


def _parse_timestamp(raw: str) -> datetime.datetime | None:
    raw = raw.strip().rstrip("Z").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.datetime.strptime(raw, fmt).replace(
                tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


def read_scene(image_path: str, meta_path: str) -> Scene:
    """Parse a ``gdalinfo``-style sidecar into a :class:`Scene`.

    Two things are handled more carefully than in the original loader:
    the UTM hemisphere is taken from the ``PROJCRS`` name (a southern-hemisphere
    scene otherwise lands in the wrong hemisphere and drags the sun with it),
    and the ``epsg=`` tag -- which can be stale in these sidecars -- is ignored
    in favour of the projection block.
    """
    with open(meta_path) as fh:
        text = fh.read()

    size = re.search(r"Size is\s+(\d+),\s*(\d+)", text)
    origin = re.search(r"Origin =\s*\(\s*([-\d.]+),\s*([-\d.]+)\)", text)
    pixel = re.search(r"Pixel Size =\s*\(\s*([-\d.]+),\s*([-\d.]+)\)", text)
    crs = re.search(r'PROJCRS\["[^"]*UTM zone (\d+)([NS])', text)
    if not (size and origin and pixel and crs):
        raise ValueError(f"{meta_path}: missing Size / Origin / Pixel Size / PROJCRS")

    nodata = re.search(r"NoData Value=([-\d.]+)", text)

    tags: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s{2}([A-Za-z_][\w]*)=(.*)$", line)
        if m:
            tags[m.group(1)] = m.group(2).strip()

    return Scene(
        image_path=image_path,
        meta_path=meta_path,
        width=int(size.group(1)),
        height=int(size.group(2)),
        origin=(float(origin.group(1)), float(origin.group(2))),
        pixel_size=(float(pixel.group(1)), float(pixel.group(2))),
        zone_number=int(crs.group(1)),
        northern=crs.group(2) == "N",
        nodata=float(nodata.group(1)) if nodata else None,
        tags=tags,
    )


def satellite_angles(scene: Scene, band: str = "B02") -> tuple[float, float, str]:
    """Satellite azimuth and zenith (degrees) for one spectral band."""
    azi = ast.literal_eval(scene.tags["satellite_azimuth"])
    zen = ast.literal_eval(scene.tags["satellite_zenith"])
    if band not in azi:
        band = sorted(azi)[0]
    return float(azi[band]), float(zen[band]), band


def sun_angles(scene: Scene, mode: str = "auto") -> dict:
    """Sun azimuth and altitude at acquisition, with a sanity check.

    ``granule_date`` is the acquisition time the original code trusts, but some
    sidecars carry a granule stamp that disagrees with the ``date`` tag by
    hours -- enough to put the sun on the wrong side of the sky and send every
    modelled shadow the wrong way.  In ``auto`` mode we keep ``granule_date``
    unless it places the sun below 10 degrees while ``date`` places it higher.
    """
    lat, lon = scene.center_latlon
    candidates: dict[str, dict] = {}
    for key in ("granule_date", "date"):
        stamp = _parse_timestamp(scene.tags.get(key, ""))
        if stamp is None:
            continue
        candidates[key] = {
            "timestamp": stamp.strftime("%Y-%m-%d %H:%M:%SZ"),
            "azimuth": float(get_azimuth(lat, lon, stamp)),
            "altitude": float(get_altitude(lat, lon, stamp)),
        }
    if not candidates:
        raise ValueError(f"{scene.meta_path}: no usable acquisition timestamp")

    if mode in candidates:
        chosen = mode
    elif mode != "auto":
        chosen = next(iter(candidates))
    else:
        chosen = "granule_date" if "granule_date" in candidates else "date"
        alt = candidates[chosen]["altitude"]
        other = "date" if chosen == "granule_date" else "granule_date"
        if alt < 10.0 and other in candidates and candidates[other]["altitude"] > alt:
            chosen = other

    result = dict(candidates[chosen])
    result["source"] = chosen
    result["candidates"] = candidates
    return result


# --------------------------------------------------------------------------
# imagery
# --------------------------------------------------------------------------

def load_image(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Return (float image with no-data filled, boolean valid mask)."""
    im = tifffile.imread(scene.image_path)
    if im.ndim > 2:
        im = im[:, :, 0]
    im = im.astype(np.float64)

    valid = np.ones(im.shape, dtype=bool)
    if scene.nodata is not None:
        valid &= im != scene.nodata
    if valid.all():
        return im, valid
    # A no-data value of 65535 would dominate every local comparison, so the
    # holes are filled with the scene median before the detector sees them.
    im = im.copy()
    im[~valid] = float(np.median(im[valid]))
    return im, valid


def run_detector(image: np.ndarray, sun: dict, sat_azimuth: float,
                 sat_zenith: float, heights: list[float],
                 diff_s: int, diff_h: int) -> np.ndarray:
    """-log10(NFA) map, straight from ``utils_eol_compute.main_NFA``."""
    return uec.main_NFA(
        "S2", image,
        sun["azimuth"], sun["altitude"],
        sat_azimuth, sat_zenith,
        nch=N_HUB_NEIGHBOURS, H=list(heights), r=RESOLUTION_M,
        sr=SAMPLE_RATE_M, ds=SHADOW_NEIGH_M, inter="bilin",
        dh=HUB_NEIGH_M, diff_s=diff_s, diff_h=diff_h,
    )


def cluster_detections(nfa: np.ndarray, valid: np.ndarray, scene: Scene,
                       threshold: float) -> list[dict]:
    """Group above-threshold pixels into one candidate per turbine."""
    mask = (nfa > threshold) & valid
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []

    peaks = ndimage.maximum_position(nfa, labels, range(1, n + 1))
    scores = ndimage.maximum(nfa, labels, range(1, n + 1))
    sizes = ndimage.sum(mask, labels, range(1, n + 1))

    out = []
    for i, ((row, col), score, size) in enumerate(zip(peaks, np.atleast_1d(scores),
                                                      np.atleast_1d(sizes))):
        lat, lon = scene.pixel_to_latlon(row, col)
        east, north = scene.pixel_to_utm(row, col)
        out.append({
            "id": i + 1,
            "row": int(row),
            "col": int(col),
            "nfa": round(float(score), 3),
            "pixels": int(size),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "easting": round(east, 1),
            "northing": round(north, 1),
        })
    out.sort(key=lambda d: -d["nfa"])
    for i, d in enumerate(out):
        d["id"] = i + 1
    return out


# --------------------------------------------------------------------------
# geometry the browser draws
# --------------------------------------------------------------------------

def sampling_geometry(sun: dict, sat_azimuth: float, sat_zenith: float,
                      height: float) -> dict:
    """The exact sample offsets the detector tested, in pixel units.

    Reusing ``main_shadow_S2`` / ``main_hub_S2`` means the overlay in the
    browser is the model the algorithm actually ran, not a redrawing of it.
    """
    tab_s = uec.main_shadow_S2(sun["azimuth"], sun["altitude"], h=height,
                               resolution=RESOLUTION_M, sample_rate=SAMPLE_RATE_M,
                               dist_shadow=SHADOW_NEIGH_M)
    shadow = [
        {"row": float(r), "col": float(c),
         "n1": [float(r1), float(c1)], "n2": [float(r2), float(c2)]}
        for r1, r, r2, c1, c, c2 in tab_s
    ]

    tab_h = uec.main_hub_S2(sat_azimuth, sat_zenith, h=height,
                            resolution=RESOLUTION_M, sample_rate=SAMPLE_RATE_M,
                            nangles=N_HUB_NEIGHBOURS)
    hub = [{"row": float(r), "col": float(c)} for c, r in tab_h]

    scale = HUB_NEIGH_M / RESOLUTION_M
    ring = [[scale * math.cos(2 * math.pi * i / N_HUB_NEIGHBOURS),
             scale * math.sin(2 * math.pi * i / N_HUB_NEIGHBOURS)]
            for i in range(N_HUB_NEIGHBOURS)]

    return {
        "height": height,
        "shadow": shadow,
        "hub": hub,
        "hubRing": ring,
        "shadowLengthM": round(height / math.tan(math.radians(sun["altitude"])), 1),
        "hubOffsetM": round(height * math.tan(math.radians(sat_zenith)), 1),
        "samples": len(shadow) + len(hub),
    }


# --------------------------------------------------------------------------
# raster packing
# --------------------------------------------------------------------------

def _data_uri(img: Image.Image, fast: bool = False) -> str:
    """Lossless WebP data URI -- about a third smaller than PNG here.

    ``fast`` trades roughly 8% of the file size for a tenth of the encoding
    time, which is what the live endpoint wants: it answers over a socket, not
    into a downloaded page.
    """
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True, quality=100, method=1 if fast else 4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pack16(values16: np.ndarray, valid: np.ndarray) -> Image.Image:
    """16-bit field as high byte / low byte / validity mask."""
    rgb = np.dstack([
        (values16 >> 8).astype(np.uint8),
        (values16 & 0xFF).astype(np.uint8),
        np.where(valid, 255, 0).astype(np.uint8),
    ])
    return Image.fromarray(rgb, "RGB")


def encode_reflectance(image: np.ndarray, valid: np.ndarray,
                       low_pct: float = 0.5, high_pct: float = 99.9,
                       gamma: float = 1.6) -> dict:
    """Raw digital numbers, losslessly, so the browser can show real values.

    Eight bits would be enough to *look* at the band, but the shadow and hub
    tests compare digital numbers against thresholds in the same units, so the
    inspector needs the original values, not a stretched copy.  The grey image
    is stretched client-side from these numbers instead.
    """
    dn = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
    lo, hi = np.percentile(image[valid], [low_pct, high_pct])
    # A 1-99 stretch blows out exactly the pixels that matter: a nacelle is one
    # of the brightest things in a grassland scene.  Reaching to the 99.9th
    # percentile keeps that headroom, and the gamma puts the midtones back.
    return {
        "uri": _data_uri(_pack16(dn, valid)),
        "min": int(image[valid].min()),
        "max": int(image[valid].max()),
        "stretch": [round(float(lo), 1), round(float(hi), 1)],
        "gamma": gamma,
    }


def encode_nfa(nfa: np.ndarray, valid: np.ndarray, fast: bool = False) -> dict:
    """-log10(NFA) packed to 16 bits over an integer value range.

    Eight bits would quantise the threshold slider into visible steps, so the
    value is split high-byte/low-byte across two channels and reassembled in
    the browser; the blue channel carries the valid mask.
    """
    sub = nfa[valid] if valid.any() else nfa
    lo = float(np.floor(np.min(sub)))
    hi = float(np.ceil(np.max(sub)))
    if hi <= lo:
        hi = lo + 1.0
    q16 = np.rint(np.clip((nfa - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)
    return {"uri": _data_uri(_pack16(q16, valid), fast), "low": lo, "high": hi,
            "max": round(float(np.max(sub)), 3)}


def nfa_histogram(nfa: np.ndarray, valid: np.ndarray, bins: int = 72) -> dict:
    counts, edges = np.histogram(nfa[valid], bins=bins)
    return {"counts": counts.tolist(),
            "edges": [round(float(e), 4) for e in edges]}


# --------------------------------------------------------------------------
# the numbers behind one pixel's score
# --------------------------------------------------------------------------

def detector_statistics(image: np.ndarray, sun: dict, sat_azimuth: float,
                        sat_zenith: float, height: float, diff_s: int,
                        diff_h: int, n_heights: int) -> dict:
    """Background probability and test count, mirroring ``main_NFA``.

    ``main_NFA`` computes these internally and returns only the map, so the
    lines below deliberately repeat its probability block -- they are what lets
    the page re-derive a single pixel's NFA from its own pixel values instead of
    just reading a number off a raster.  Note ``diff_h=0`` in the hub
    probability: that is what the published code does, and changing it here
    would make the page disagree with the map beside it.
    """
    tab_s = uec.main_shadow_S2(sun["azimuth"], sun["altitude"], h=height,
                               resolution=RESOLUTION_M, sample_rate=SAMPLE_RATE_M,
                               dist_shadow=SHADOW_NEIGH_M)
    rrN1, rr, rrN2, ccN1, cc, ccN2 = (tab_s[:, i] for i in range(6))
    tab_h = uec.main_hub_S2(sat_azimuth, sat_zenith, h=height,
                            resolution=RESOLUTION_M, sample_rate=SAMPLE_RATE_M,
                            nangles=N_HUB_NEIGHBOURS)
    hub_cc, hub_rr = tab_h[:, 0], tab_h[:, 1]

    nk_shadow, nk_hub = len(rr), len(hub_rr)
    k_max = nk_shadow + nk_hub

    p_shadow = float(uec.calc_p_emp_shadow(image, [rr[0], cc[0]],
                                           [rrN1[0], ccN1[0]], diff_s))
    p_hub = float(uec.calc_p_emp_hub(image, RESOLUTION_M, HUB_NEIGH_M,
                                     nch=N_HUB_NEIGHBOURS, diff_h=0))
    pmoy = (nk_shadow * p_shadow + nk_hub * p_hub) / k_max

    nrow, ncol = image.shape
    margin_x = max(np.abs(np.concatenate((rr, rrN1, rrN2))))
    margin_y = max(np.abs(np.concatenate((cc, ccN1, ccN2))))
    hub_margin = HUB_NEIGH_M / RESOLUTION_M
    ecx = int(np.ceil(max(margin_x, hub_margin + max(abs(hub_rr)))))
    ecy = int(np.ceil(max(margin_y, hub_margin + max(abs(hub_cc)))))
    ntests = (nrow - 2 * ecx - 1) * (ncol - 2 * ecy - 1)

    return {
        "pShadow": p_shadow,
        "pHub": p_hub,
        "pMean": pmoy,
        "kMax": int(k_max),
        "nShadow": int(nk_shadow),
        "nHub": int(nk_hub),
        "nTests": int(ntests),
        "nHeights": int(n_heights),
        "marginRows": ecx,
        "marginCols": ecy,
    }
