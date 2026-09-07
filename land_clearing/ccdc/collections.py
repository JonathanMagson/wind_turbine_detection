"""Harmonised Sentinel-1 and Sentinel-2 feature stacks for CCDC.

CCDC fits a harmonic season + linear trend model per pixel per band and flags a
break when consecutive residuals exceed that pixel's own fitted RMSE. That makes
the *inputs* matter more than usual:

  * Every band in ``breakpointBands`` must be present in every image, because
    CCDC drops an observation entirely if any breakpoint band is masked there.
    Sentinel-1 and Sentinel-2 have complementary gaps (orbit cadence vs cloud),
    so they are built as two separate stacks and run through CCDC separately.
    Their breaks are reconciled afterwards in ``rules.py``. Interleaving them
    into one collection would throw away every radar observation that happened
    to fall under a cloud.
  * Band magnitudes should be comparable within a stack, because CCDC's
    ``lambda`` regularisation is a single scalar across bands. Both stacks are
    scaled to roughly the 0-10000 range that CCDC's defaults assume.
  * Speckle inflates the fitted RMSE, which raises the break threshold and
    costs sensitivity. Sentinel-1 is multi-looked before stacking.

Scaling, which the magnitude thresholds in ``rules.py`` are expressed in:
  Sentinel-1  dB x 100      (VH of -18 dB -> -1800; a 2 dB drop -> -200)
  Sentinel-2  index x 10000 (NDVI of 0.62 -> 6200; a 0.2 drop -> -2000)
"""

import ee

S1_BANDS = ['VV', 'VH', 'RATIO']
S2_BANDS = ['NDVI', 'NBR', 'BSI']

S1_SCALE = 100      # dB -> integer-ish units
S2_SCALE = 10000    # index -> reflectance-like units


def _multilook(image, kernel_px):
    """Boxcar multi-look in linear power, returned in dB.

    Averaging must happen in linear power, not dB, or the result is biased low.
    """
    if not kernel_px or kernel_px <= 1:
        return image
    names = image.bandNames()
    linear = ee.Image(10).pow(image.divide(10))
    smoothed = linear.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(radius=kernel_px // 2, units='pixels'))
    return smoothed.log10().multiply(10).rename(names)


def s1_stack(aoi, start, end, orbit_pass='DESCENDING', relative_orbit=None,
             multilook_px=3):
    """Sentinel-1 collection with VV, VH and the VH/VV ratio, all in dB x 100.

    Returns ``(collection, relative_orbit)``. Restricted to a single relative
    orbit by default: local incidence angle varies between tracks, and CCDC
    would read the resulting step as a break. Pass ``relative_orbit=0`` to
    disable the restriction and accept the noise.
    """
    coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(ee.Date(start), ee.Date(end))
            .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH']))
            .filter(ee.Filter.eq('resolution_meters', 10))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass)))

    if relative_orbit is None:
        orbits = coll.aggregate_array('relativeOrbitNumber_start').getInfo() or []
        if not orbits:
            raise ValueError(
                'No Sentinel-1 images over the AOI for these filters. Try the '
                'other orbit pass or a wider date range.')
        counts = {}
        for o in orbits:
            counts[int(o)] = counts.get(int(o), 0) + 1
        relative_orbit = max(counts.items(), key=lambda kv: kv[1])[0]
    if relative_orbit:
        coll = coll.filter(
            ee.Filter.eq('relativeOrbitNumber_start', relative_orbit))

    def prep(image):
        db = _multilook(image.select(['VV', 'VH']), multilook_px)
        # In dB the ratio is a difference, which is also the log of VH/VV.
        ratio = db.select('VH').subtract(db.select('VV')).rename('RATIO')
        out = (db.addBands(ratio).multiply(S1_SCALE)
               .rename(S1_BANDS)
               .clip(aoi))
        # copyProperties returns an Element, not an Image; cast it back so the
        # mapped collection stays a well-typed ImageCollection.
        return ee.Image(out.copyProperties(image, ['system:time_start']))

    return coll.sort('system:time_start').map(prep), relative_orbit


def _mask_s2(image):
    """Scene classification cloud/shadow/snow mask for S2_SR_HARMONIZED."""
    scl = image.select('SCL')
    # 3 shadow, 8/9 cloud medium+high, 10 cirrus, 11 snow.
    bad = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9))
           .Or(scl.eq(10)).Or(scl.eq(11)))
    return image.updateMask(bad.Not())


def s2_stack(aoi, start, end, max_cloud_pct=60):
    """Sentinel-2 collection with NDVI, NBR and a bare-soil index, x 10000.

    NDVI carries green cover, NBR carries the char/structure signal that helps
    separate fire from clearing, and BSI carries exposed soil -- the cue that
    makes grassland conversion legible at all, since grassland has no volume
    scattering for the radar to lose.
    """
    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(ee.Date(start), ee.Date(end))
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
            .map(_mask_s2))

    def prep(image):
        r = image.select(['B2', 'B4', 'B8', 'B11'],
                         ['blue', 'red', 'nir', 'swir1']).divide(10000)
        ndvi = r.normalizedDifference(['nir', 'red']).rename('NDVI')
        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        # Bare Soil Index (Rikimaru et al. 2002), bounded to roughly [-1, 1].
        bsi = r.expression(
            '((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))',
            {'swir1': r.select('swir1'), 'red': r.select('red'),
             'nir': r.select('nir'), 'blue': r.select('blue')}).rename('BSI')
        out = (ndvi.addBands(nbr).addBands(bsi).multiply(S2_SCALE)
               .rename(S2_BANDS)
               .clip(aoi))
        return ee.Image(out.copyProperties(image, ['system:time_start']))

    return coll.sort('system:time_start').map(prep)
