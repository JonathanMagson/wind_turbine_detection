"""Parse Sentinel-1 GRD annotation: geolocation grid and calibration LUT.

A GRD product is in ground-range/azimuth geometry with no map projection. Two
small XML files per polarisation carry what is needed to use it:

  ``annotation/iw-<pol>.xml``
      A geolocation grid of tie points, each giving (line, pixel) -> (lat, lon).
      Inverting it gives the pixel window for an AOI.

  ``annotation/calibration/calibration-iw-<pol>.xml``
      A sigma-nought LUT on a coarse (line, pixel) grid. Calibrated backscatter
      is ``sigma0 = DN^2 / sigmaNought^2``.

Both are around 1-2 MB, against ~600 MB for the measurement raster, so the
annotation is cheap to fetch in full.

Inverting the geolocation grid with a single polynomial over the whole slice
(300 x 170 km) plateaus around 110 m error in the range direction, because the
ground-range mapping is not well approximated globally. Restricting the fit to
tie points near the AOI and using a cubic gets it under a pixel; that is what
``GeoGrid`` does.
"""

import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from . import http_util

BUCKET = 'https://sentinel-s1-l1c.s3.amazonaws.com'

# Radius in degrees around the AOI centre for selecting tie points. Wide enough
# to constrain a cubic, narrow enough to stay in the locally-smooth regime.
FIT_RADIUS_DEG = 0.35
FIT_DEGREE = 3


def fetch(scene_path, rel, timeout=120):
    return http_util.fetch(f'{BUCKET}/{scene_path}/{rel}', timeout=timeout)


def _design(lat, lon, degree):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    return np.column_stack([(lat ** i) * (lon ** j)
                            for i in range(degree + 1)
                            for j in range(degree + 1 - i)])


class GeoGrid:
    """Maps (lon, lat) to (line, pixel) for one scene, fitted near an AOI."""

    def __init__(self, xml_bytes, centre_lon, centre_lat,
                 radius=FIT_RADIUS_DEG, degree=FIT_DEGREE):
        root = ET.fromstring(xml_bytes)
        info = root.find('.//imageAnnotation/imageInformation')
        self.height = int(info.find('numberOfLines').text)
        self.width = int(info.find('numberOfSamples').text)

        pts = root.findall('.//geolocationGridPointList/geolocationGridPoint')
        grid = np.array([[float(p.find(k).text)
                          for k in ('line', 'pixel', 'latitude', 'longitude')]
                         for p in pts])

        # Widen the neighbourhood until the fit is well determined.
        n_terms = (degree + 1) * (degree + 2) // 2
        while True:
            sel = ((np.abs(grid[:, 2] - centre_lat) < radius)
                   & (np.abs(grid[:, 3] - centre_lon) < radius))
            if sel.sum() > n_terms + 4 or radius > 4.0:
                break
            radius *= 1.4
        if sel.sum() <= n_terms:
            raise ValueError('Too few geolocation tie points near the AOI.')

        lat, lon = grid[sel, 2], grid[sel, 3]
        A = _design(lat, lon, degree)
        self._degree = degree
        self._c_line, *_ = np.linalg.lstsq(A, grid[sel, 0], rcond=None)
        self._c_pixel, *_ = np.linalg.lstsq(A, grid[sel, 1], rcond=None)
        self.rms_line = float(np.sqrt(((A @ self._c_line - grid[sel, 0]) ** 2).mean()))
        self.rms_pixel = float(np.sqrt(((A @ self._c_pixel - grid[sel, 1]) ** 2).mean()))
        self.n_points = int(sel.sum())

    def to_line_pixel(self, lon, lat):
        """(lon, lat) arrays -> (line, pixel) float arrays."""
        A = _design(lat, lon, self._degree)
        return A @ self._c_line, A @ self._c_pixel


class CalibrationLUT:
    """Sigma-nought LUT, bilinearly interpolated in (line, pixel)."""

    def __init__(self, xml_bytes):
        root = ET.fromstring(xml_bytes)
        vectors = root.findall('.//calibrationVectorList/calibrationVector')
        if not vectors:
            raise ValueError('No calibration vectors in annotation.')
        self.lines = np.array([float(v.find('line').text) for v in vectors])
        self.pixels = np.array(
            [float(x) for x in vectors[0].find('pixel').text.split()])
        self.sigma = np.array(
            [[float(x) for x in v.find('sigmaNought').text.split()]
             for v in vectors])

    def sigma_nought(self, line, pixel):
        """Interpolate the LUT at float (line, pixel) arrays."""
        li = np.interp(line, self.lines, np.arange(self.lines.size))
        pi = np.interp(pixel, self.pixels, np.arange(self.pixels.size))
        l0 = np.clip(np.floor(li).astype(int), 0, self.sigma.shape[0] - 2)
        p0 = np.clip(np.floor(pi).astype(int), 0, self.sigma.shape[1] - 2)
        dl, dp = li - l0, pi - p0
        s = self.sigma
        return ((s[l0, p0] * (1 - dl) * (1 - dp))
                + (s[l0 + 1, p0] * dl * (1 - dp))
                + (s[l0, p0 + 1] * (1 - dl) * dp)
                + (s[l0 + 1, p0 + 1] * dl * dp))
