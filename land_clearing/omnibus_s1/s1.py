"""Build an omnibus-ready Sentinel-1 time series over an AOI.

The sequential omnibus test assumes every image in the series is an independent
realisation of the same imaging geometry. In practice that means:

  * one instrument mode (IW), one polarisation pair (VV+VH), 10 m GRD;
  * one orbit pass (ASCENDING or DESCENDING) *and* one relative orbit number,
    so the local incidence angle is constant per pixel;
  * the AOI fully contained in every footprint (partial overlaps break the
    per-pixel series length);
  * linear power, not dB, scaled by the equivalent number of looks.

GEE's COPERNICUS/S1_GRD is calibrated to sigma-nought and geometrically terrain
corrected, but not radiometrically terrain flattened. That is acceptable here
because the test is purely temporal and per-pixel, but it does mean steep
terrain will have noisier series -- see README.md.
"""

import math

import ee

# Equivalent number of looks for the GEE S1 GRD product, as used in the
# Earth Engine community tutorial.
ENL = 4.4


def s1_collection(aoi, start, end, orbit_pass='DESCENDING'):
    """Filtered COPERNICUS/S1_GRD collection fully containing ``aoi``."""
    coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(ee.Date(start), ee.Date(end))
            .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH']))
            .filter(ee.Filter.eq('resolution_meters', 10))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass)))
    return coll.filter(ee.Filter.contains(rightValue=aoi, leftField='.geo'))


def relative_orbit_counts(coll):
    """Client-side {relative_orbit: image_count} for a filtered collection."""
    orbits = coll.aggregate_array('relativeOrbitNumber_start').getInfo() or []
    counts = {}
    for o in orbits:
        o = int(o)
        counts[o] = counts.get(o, 0) + 1
    return counts


def best_relative_orbit(coll):
    """Relative orbit with the most images. Raises if the collection is empty."""
    counts = relative_orbit_counts(coll)
    if not counts:
        raise ValueError(
            'No Sentinel-1 images fully contain the AOI for these filters. '
            'Try the other orbit pass, a wider date range, or a smaller AOI.')
    return max(counts.items(), key=lambda kv: kv[1])[0]


def to_linear_power(image):
    """dB sigma0 -> linear power, keeping only VV and VH."""
    out = (image.select('VV', 'VH')
           .multiply(ee.Image.constant(math.log(10.0) / 10.0))
           .exp())
    # copyProperties returns an Element, not an Image; cast it back.
    return ee.Image(out.copyProperties(image, ['system:time_start']))


def build_series(aoi, start, end, orbit_pass='DESCENDING', relative_orbit=None,
                 stride=1, enl=ENL):
    """Return ``(im_list, timestamps, relative_orbit)`` ready for ``change_maps``.

    ``im_list`` is an ``ee.List`` of clipped linear-power images scaled by
    ``enl``. ``timestamps`` is a client-side list of ISO date strings, one per
    image, in acquisition order; interval *i* of the resulting ``bmap`` spans
    ``timestamps[i-1] -> timestamps[i]``.

    ``stride`` subsamples the series (stride=2 keeps every second acquisition),
    which is a cheap way to cut compute when a 6-day revisit is more than the
    application needs.
    """
    coll = s1_collection(aoi, start, end, orbit_pass)
    if relative_orbit is None:
        relative_orbit = best_relative_orbit(coll)
    coll = (coll.filter(ee.Filter.eq('relativeOrbitNumber_start', relative_orbit))
            .sort('system:time_start'))

    all_stamps = coll.aggregate_array('system:time_start').getInfo() or []
    keep = list(range(0, len(all_stamps), stride))
    stamps = [all_stamps[i] for i in keep]
    if len(stamps) < 3:
        raise ValueError(
            'Only %d acquisitions after filtering; the omnibus test needs at '
            'least 3 (and realistically 15+) to be useful.' % len(stamps))

    images = coll.map(to_linear_power).toList(len(all_stamps))
    im_list = ee.List(keep).map(
        lambda i: ee.Image(images.get(i)).multiply(enl).clip(aoi))

    # One server-side call for all dates rather than one getInfo per image.
    dates = ee.List(stamps).map(
        lambda t: ee.Date(t).format('YYYY-MM-dd')).getInfo()

    return im_list, dates, relative_orbit
