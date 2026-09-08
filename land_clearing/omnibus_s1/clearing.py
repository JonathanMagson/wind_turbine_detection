"""Turn a generic omnibus change map into a land-clearing product.

The omnibus test is cover-agnostic: it flags any significant change in the
polarimetric signature. Three domain constraints make it a clearing product.
Note that these constraints are what restrict this pipeline to woody
vegetation -- for a cover-agnostic product see ``land_clearing.ccdc``.

  1. Direction. Removing woody vegetation destroys volume scattering, so
     backscatter drops. Only negative-definite changes (bmap == 2) are kept.
  2. Baseline woody cover. Clearing requires something to clear. Masking to a
     woody baseline removes most crop-cycle and soil-moisture false positives.
  3. Minimum mapping unit. Single 10 m pixels are noise; NSW SLATS reports
     woody change at roughly 0.1-0.5 ha.
"""

import ee


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
