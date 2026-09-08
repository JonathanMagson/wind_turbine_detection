"""Read and geocode a small AOI window from a Sentinel-1 GRD scene on S3.

The measurement rasters are ~600 MB but internally tiled at 1024x1024, so GDAL
can range-read a small window over HTTPS in about a second. Nothing is
downloaded in full and nothing is written to disk.

Output is resampled onto a regular EPSG:4326 grid defined by the AOI, so scenes
from different dates land on the same grid and are co-registered by
construction (to the accuracy of each scene's own geolocation grid, which
``annotation.GeoGrid`` reports).

Two things this deliberately does not do, both of which matter for
interpretation rather than for running:
  * No radiometric terrain flattening. Sigma-nought here is ellipsoid
    referenced, so slopes are biased. Fine over flat inland NSW, not over the
    ranges.
  * No precise orbit files. Geolocation comes from the tie points shipped with
    the product, which is metre-level for our purposes but not survey grade.
"""

import os

import numpy as np

_GDAL_ENV_READY = False


def _prepare_gdal_env():
    """Point GDAL at the session proxy and CA bundle, and stop it listing dirs."""
    global _GDAL_ENV_READY
    if _GDAL_ENV_READY:
        return
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if proxy and not os.environ.get('GDAL_HTTP_PROXY'):
        os.environ['GDAL_HTTP_PROXY'] = proxy
    ca = '/root/.ccr/ca-bundle.crt'
    if os.path.exists(ca):
        os.environ.setdefault('GDAL_HTTP_CAINFO', ca)
        os.environ.setdefault('CURL_CA_BUNDLE', ca)
    # Without this GDAL issues a directory listing per open, which is slow and
    # pointless against a bucket prefix holding a 600 MB raster.
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tiff,.tif')
    _GDAL_ENV_READY = True


def aoi_grid(bbox, pixel_deg):
    """Regular lon/lat mesh for ``bbox`` = (west, south, east, north)."""
    w, s, e, n = bbox
    lons = np.arange(w, e, pixel_deg)
    lats = np.arange(n, s, -pixel_deg)
    return np.meshgrid(lons, lats)


def read_geocoded(scene_path, pol, bbox, pixel_deg, geo, cal, margin=64):
    """Sigma-nought for one polarisation, resampled onto the AOI grid.

    Returns a float32 array shaped like the AOI grid, with NaN where the AOI
    falls outside the scene.
    """
    _prepare_gdal_env()
    import rasterio
    from rasterio.windows import Window
    from scipy.ndimage import map_coordinates

    lon_g, lat_g = aoi_grid(bbox, pixel_deg)
    line, pixel = geo.to_line_pixel(lon_g.ravel(), lat_g.ravel())
    line = line.reshape(lon_g.shape)
    pixel = pixel.reshape(lon_g.shape)

    r0 = int(np.floor(line.min())) - margin
    r1 = int(np.ceil(line.max())) + margin
    c0 = int(np.floor(pixel.min())) - margin
    c1 = int(np.ceil(pixel.max())) + margin
    if r0 < 0 or c0 < 0 or r1 > geo.height or c1 > geo.width:
        raise ValueError(
            'AOI falls outside scene %s (rows %d..%d of %d, cols %d..%d of %d). '
            'Slice footprints are rotated, so an AOI inside the bounding box can '
            'still fall off the edge -- try a smaller or more central AOI.'
            % (scene_path.split('/')[-1], r0, r1, geo.height, c0, c1, geo.width))

    url = f'/vsicurl/https://sentinel-s1-l1c.s3.amazonaws.com/{scene_path}/measurement/iw-{pol}.tiff'
    with rasterio.open(url) as ds:
        dn = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)

    # Resample DN onto the AOI grid, then calibrate. Calibrating first would
    # mean interpolating the LUT over the whole window rather than only where
    # output samples land.
    dn_at = map_coordinates(dn, [line - r0, pixel - c0], order=1,
                            mode='constant', cval=np.nan)
    sigma_nought = cal.sigma_nought(line, pixel)
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma0 = (dn_at ** 2) / (sigma_nought ** 2)
    sigma0[~np.isfinite(sigma0)] = np.nan
    return sigma0.astype(np.float32)


def read_scene(scene, bbox, pixel_deg, pols=('vv', 'vh')):
    """Dual-pol sigma-nought stack for one scene, shaped (2, H, W).

    ``scene`` is a productInfo dict from ``catalog``.
    """
    from . import annotation

    path = scene['path']
    w, s, e, n = bbox
    clon, clat = (w + e) / 2.0, (s + n) / 2.0

    out, diag = [], {}
    for pol in pols:
        geo = annotation.GeoGrid(
            annotation.fetch(path, f'annotation/iw-{pol}.xml'), clon, clat)
        cal = annotation.CalibrationLUT(
            annotation.fetch(path, f'annotation/calibration/calibration-iw-{pol}.xml'))
        out.append(read_geocoded(path, pol, bbox, pixel_deg, geo, cal))
        diag[pol] = {'rms_line_px': round(geo.rms_line, 3),
                     'rms_pixel_px': round(geo.rms_pixel, 3),
                     'tie_points': geo.n_points}
    return np.stack(out), diag
