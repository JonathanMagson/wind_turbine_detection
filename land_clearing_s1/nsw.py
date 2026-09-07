"""NSW-specific AOIs, woody-vegetation masks and land-clearing post-processing.

The omnibus test is a generic change detector: it flags *any* significant change
in the polarimetric signature. Turning that into a land-clearing product needs
three domain constraints, applied here:

  1. Direction. Removing woody vegetation destroys volume scattering, so
     backscatter (VH especially) drops. Only negative-definite changes
     (bmap == 2) are kept.
  2. Baseline woody cover. Clearing can only happen where there was woody
     vegetation to start with, which removes most crop-cycle and soil-moisture
     false positives on cultivated paddocks.
  3. Minimum mapping unit. Single-pixel detections at 10 m are noise; NSW SLATS
     reports woody change at roughly 0.1-0.5 ha, so a connected-component
     filter is applied at a comparable scale.
"""

import ee

# Bounding boxes as (west, south, east, north) in EPSG:4326.
# Chosen in NSW landscapes where SLATS has repeatedly reported woody clearing.
AOIS = {
    # Brigalow Belt South / north-west slopes cropping frontier.
    'moree': (149.60, -29.60, 150.00, -29.30),
    'walgett': (148.00, -30.20, 148.40, -29.90),
    'collarenebri': (148.40, -29.70, 148.80, -29.40),
    'narrabri': (149.55, -30.45, 149.95, -30.15),
    # Pilliga / Nandewar woodland margin.
    'pilliga': (148.90, -30.90, 149.30, -30.60),
    # Riverina mallee and box woodland edge.
    'hay': (144.60, -34.70, 145.00, -34.40),
    # South-east forestry and clearing mix.
    'bombala': (149.00, -37.00, 149.40, -36.70),
}

# Woody baselines available in the GEE public catalogue for Australia.
WOODY_MASKS = ('worldcover', 'hansen', 'palsar', 'none')


def aoi_geometry(name=None, bbox=None):
    """``ee.Geometry`` from a preset name or an explicit w,s,e,n bbox."""
    if bbox is not None:
        w, s, e, n = bbox
    elif name is not None:
        if name not in AOIS:
            raise ValueError('Unknown AOI %r. Known: %s'
                             % (name, ', '.join(sorted(AOIS))))
        w, s, e, n = AOIS[name]
    else:
        raise ValueError('Provide either an AOI name or a bbox.')
    return ee.Geometry.Rectangle([w, s, e, n])


def woody_mask(kind='worldcover', tree_cover_pct=20, year=2020):
    """Binary baseline woody-cover mask (1 = woody), or ``None`` for no mask.

    worldcover  ESA WorldCover v200 (2021) classes 10 (tree) and 20 (shrubland),
                10 m. Best spatial detail; under-maps very sparse woody cover,
                which is common in western NSW.
    hansen      Hansen GFC v1.11 ``treecover2000`` above ``tree_cover_pct``,
                30 m. Well understood, but a 2000 baseline drifts, and it
                under-maps NSW woodland below ~20% canopy.
    palsar      JAXA ALOS/PALSAR yearly forest/non-forest (FNF4) for ``year``,
                25 m, classes 1-2 (dense + non-dense forest). Radar-derived, so
                its notion of woody is closest to what S1 actually senses.
    none        No mask; every negative change is kept. Useful for seeing the
                raw false-positive rate.
    """
    if kind == 'none':
        return None
    if kind == 'worldcover':
        wc = ee.Image('ESA/WorldCover/v200/2021').select('Map')
        return wc.eq(10).Or(wc.eq(20)).rename('woody')
    if kind == 'hansen':
        gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
        return gfc.select('treecover2000').gte(tree_cover_pct).rename('woody')
    if kind == 'palsar':
        fnf = (ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/FNF4')
               .filterDate('%d-01-01' % year, '%d-01-01' % (year + 1))
               .first())
        if fnf is None:
            raise ValueError('No PALSAR FNF4 mosaic for year %d.' % year)
        fnf = ee.Image(fnf).select('fnf')
        return fnf.eq(1).Or(fnf.eq(2)).rename('woody')
    raise ValueError('Unknown woody mask %r. Known: %s'
                     % (kind, ', '.join(WOODY_MASKS)))


def clearing_from_bmap(bmap, n_intervals, mask=None, min_mmu_ha=0.5,
                       scale=10, max_size=1024):
    """Reduce an omnibus ``bmap`` to a land-clearing product.

    Returns an ``ee.Image`` with bands:
      clearing        1 where clearing was detected, else 0
      first_interval  1-based index of the first interval with a negative
                      change (0 where none); map it back to dates with the
                      ``dates`` list returned by ``s1.build_series``
      n_negative      count of negative-definite changes over the series
    """
    bmap = ee.Image(bmap)

    negative = bmap.eq(2)
    n_negative = negative.reduce(ee.Reducer.sum()).rename('n_negative')

    # Earliest interval wins, so walk backwards.
    first = ee.Image(0).rename('first_interval')
    for i in range(n_intervals - 1, -1, -1):
        first = first.where(negative.select(i), i + 1)

    clearing = first.gt(0).rename('clearing')
    if mask is not None:
        clearing = clearing.And(ee.Image(mask).unmask(0))

    if min_mmu_ha:
        min_pixels = int(round(min_mmu_ha * 10000.0 / (scale * scale)))
        if min_pixels > 1:
            size = (clearing.selfMask()
                    .connectedPixelCount(maxSize=max_size, eightConnected=True))
            clearing = clearing.And(size.unmask(0).gte(min_pixels))

    clearing = clearing.rename('clearing')
    return clearing.addBands(first.multiply(clearing).rename('first_interval')) \
                   .addBands(n_negative.multiply(clearing).rename('n_negative')) \
                   .toInt()


def cleared_area_ha(clearing, aoi, scale=10, max_pixels=1e13):
    """Total detected cleared area in hectares (client-side float)."""
    area = (ee.Image(clearing).select('clearing')
            .multiply(ee.Image.pixelArea())
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
                          maxPixels=max_pixels, bestEffort=False))
    return ee.Number(area.get('clearing')).divide(10000).getInfo()


def area_by_interval(clearing, aoi, n_intervals, scale=10, max_pixels=1e13):
    """Detected cleared hectares per interval index (client-side dict)."""
    first = ee.Image(clearing).select('first_interval')
    stats = (ee.Image.pixelArea().addBands(first)
             .reduceRegion(reducer=ee.Reducer.sum().group(groupField=1,
                                                          groupName='interval'),
                           geometry=aoi, scale=scale, maxPixels=max_pixels))
    groups = ee.List(stats.get('groups')).getInfo() or []
    out = {i: 0.0 for i in range(1, n_intervals + 1)}
    for g in groups:
        idx = int(g['interval'])
        if idx > 0:
            out[idx] = g['sum'] / 10000.0
    return out
