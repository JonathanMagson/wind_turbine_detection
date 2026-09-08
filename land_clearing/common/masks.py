"""Baseline vegetation masks and land-cover strata for NSW.

``woody_mask`` supports the omnibus pipeline, which only makes sense inside
woody vegetation. ``cover_strata`` supports the CCDC pipeline, which runs across
forest, woodland and grassland and therefore needs to know which stratum each
pixel falls in so the break rules can differ per cover type.
"""

import ee

WOODY_MASKS = ('worldcover', 'hansen', 'palsar', 'none')

# Stratum codes used throughout the CCDC pipeline.
FOREST, WOODLAND, GRASSLAND, CROPLAND = 1, 2, 3, 4
STRATUM_NAMES = {FOREST: 'forest', WOODLAND: 'woodland',
                 GRASSLAND: 'grassland', CROPLAND: 'cropland'}


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


def cover_strata(include_cropland=False):
    """Land-cover stratum image (see ``STRATUM_NAMES``), 0 outside all strata.

    Derived from ESA WorldCover v200 (2021, 10 m), the only global product at
    S1 resolution that separates tree cover, shrubland and grassland. Cropland
    is mapped but excluded by default: sown-crop cycles produce large, regular
    breaks in both radar and optical series that are not land clearing.
    """
    wc = ee.Image('ESA/WorldCover/v200/2021').select('Map')
    strata = (ee.Image(0)
              .where(wc.eq(10), FOREST)
              .where(wc.eq(20), WOODLAND)
              .where(wc.eq(30), GRASSLAND))
    if include_cropland:
        strata = strata.where(wc.eq(40), CROPLAND)
    return strata.rename('stratum').toInt()
