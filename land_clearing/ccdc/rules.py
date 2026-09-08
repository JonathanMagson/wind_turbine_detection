"""Cover-stratified rules that turn CCDC breaks into land clearing.

CCDC flags departures from a pixel's own fitted seasonal model. That is
cover-agnostic, which is the point -- but a break means different things in
different vegetation, and the evidence available to confirm it differs too.

  Forest and woodland
      Clearing destroys volume scattering, so Sentinel-1 VH falls sharply and
      unambiguously. Radar alone is sufficient evidence, and the direction
      constraint (negative magnitude) does most of the false-positive
      rejection. Optical agreement, when available, raises confidence.

  Grassland
      There is almost no volume scattering to lose. Ploughing changes surface
      roughness, and the resulting backscatter change can go either way
      depending on soil moisture, residue cover and row direction relative to
      the look azimuth -- so a radar direction rule is not defensible here.
      Optical carries the real signal: green cover falls and exposed soil
      rises. Grassland conversion therefore *requires* the Sentinel-2 evidence
      (NDVI drop and BSI rise), with radar used only as corroboration.

  Cropland
      Excluded by default. Sown-crop cycles produce large, regular breaks in
      both radar and optical that are not land clearing.

Fire is the main residual confusion in every stratum. A burn drops NDVI and
raises BSI much like clearing does, but it also drives NBR sharply negative
while clearing moves it far less. ``nbr_fire_drop`` uses that to drop
likely-burn detections; it is a first-order screen, not a substitute for a
burnt-area product.

Magnitudes are in the scaled units from ``collections.py``:
Sentinel-1 dB x 100, Sentinel-2 index x 10000.
"""

import ee

from ..common import masks

# Defaults, expressed in scaled units. Chosen from the physical size of the
# effect rather than tuned against a reference, so treat them as a starting
# point and calibrate against SLATS for your AOI.
VH_DROP_WOODY = -150      # -1.5 dB: conservative for canopy removal
VH_DROP_GRASS = -100      # -1.0 dB: corroboration only, not decisive
NDVI_DROP = -1500         # -0.15 NDVI
BSI_RISE = 800            # +0.08 BSI
NBR_FIRE_DROP = -3000     # -0.30 NBR suggests burn rather than clearing
CHANGE_PROB = 0.90


def _agreement(s1_breaks, s2_breaks, tolerance_years):
    """1 where both sensors broke within ``tolerance_years`` of each other."""
    both = s1_breaks.select('s1_n_breaks').gt(0).And(
        s2_breaks.select('s2_n_breaks').gt(0))
    gap = (s1_breaks.select('s1_t_break')
           .subtract(s2_breaks.select('s2_t_break')).abs())
    return both.And(gap.lte(tolerance_years)).rename('agreement')


def clearing_rules(s1_breaks=None, s2_breaks=None, strata=None,
                   vh_drop_woody=VH_DROP_WOODY, vh_drop_grass=VH_DROP_GRASS,
                   ndvi_drop=NDVI_DROP, bsi_rise=BSI_RISE,
                   nbr_fire_drop=NBR_FIRE_DROP, change_prob=CHANGE_PROB,
                   agreement_days=60, screen_fire=True):
    """Apply per-stratum evidence rules. Returns an unfiltered clearing image.

    Either sensor may be ``None``, in which case rules that need it are
    dropped: with Sentinel-1 only, grassland cannot be assessed at all and is
    left unflagged rather than guessed at.

    Bands returned:
      ``clearing``    1 where the stratum's rule is satisfied
      ``stratum``     cover stratum code (see ``common.masks``)
      ``t_break``     fractional year of the break that triggered it
      ``confidence``  1 single-sensor, 2 both sensors agree in time
    """
    if s1_breaks is None and s2_breaks is None:
        raise ValueError('At least one of s1_breaks or s2_breaks is required.')
    if strata is None:
        strata = masks.cover_strata()
    strata = ee.Image(strata).select('stratum')

    tolerance_years = agreement_days / 365.25
    zero = ee.Image(0)

    woody_px = strata.eq(masks.FOREST).Or(strata.eq(masks.WOODLAND))
    grass_px = strata.eq(masks.GRASSLAND)
    crop_px = strata.eq(masks.CROPLAND)

    # --- Woody: radar-led, direction-constrained. ---
    # Radar decides when it is available. Optical alone is not used here
    # because canopy NDVI also falls for drought and deciduous phenology,
    # whereas a VH collapse in woody cover is close to specific to structural
    # loss. Optical still contributes, via the confidence band and the fire
    # screen below.
    if s1_breaks is not None:
        woody_hit = (s1_breaks.select('s1_n_breaks').gt(0)
                     .And(s1_breaks.select('s1_change_prob').gte(change_prob))
                     .And(s1_breaks.select('s1_VH_mag').lte(vh_drop_woody)))
    else:
        # No radar: fall back to optical so woody is not simply lost, and
        # accept the weaker specificity.
        woody_hit = (s2_breaks.select('s2_n_breaks').gt(0)
                     .And(s2_breaks.select('s2_change_prob').gte(change_prob))
                     .And(s2_breaks.select('s2_NDVI_mag').lte(ndvi_drop)))

    # --- Grassland: optical-led, radar is corroboration only. ---
    if s2_breaks is not None:
        grass_hit = (s2_breaks.select('s2_n_breaks').gt(0)
                     .And(s2_breaks.select('s2_change_prob').gte(change_prob))
                     .And(s2_breaks.select('s2_NDVI_mag').lte(ndvi_drop))
                     .And(s2_breaks.select('s2_BSI_mag').gte(bsi_rise)))
    else:
        # Radar alone cannot establish grassland conversion; say so by
        # detecting nothing rather than emitting a number nobody should trust.
        grass_hit = zero
    if s2_breaks is not None and s1_breaks is not None:
        # Radar corroboration for grassland: a backscatter drop alongside the
        # optical break. Not required, but it feeds the confidence band.
        grass_corroborated = s1_breaks.select('s1_VH_mag').lte(vh_drop_grass)
    else:
        grass_corroborated = zero

    hit = (woody_px.And(woody_hit)).Or(grass_px.And(grass_hit))
    # Explicit, so the exclusion still holds if strata were built with
    # include_cropland=True.
    hit = hit.And(crop_px.Not())

    # --- Fire screen. ---
    if screen_fire and s2_breaks is not None:
        likely_burn = s2_breaks.select('s2_NBR_mag').lte(nbr_fire_drop)
        hit = hit.And(likely_burn.Not())

    hit = hit.rename('clearing')

    # Break date: prefer the sensor that drove the stratum's rule.
    if s1_breaks is not None and s2_breaks is not None:
        t_break = (s2_breaks.select('s2_t_break')
                   .where(woody_px, s1_breaks.select('s1_t_break')))
        agree = _agreement(s1_breaks, s2_breaks, tolerance_years)
        # In grassland, temporal agreement only counts if the radar moved in
        # the direction a conversion would produce.
        agree = agree.And(grass_px.Not().Or(grass_corroborated))
        confidence = agree.add(1)
    elif s1_breaks is not None:
        t_break = s1_breaks.select('s1_t_break')
        confidence = ee.Image(1)
    else:
        t_break = s2_breaks.select('s2_t_break')
        confidence = ee.Image(1)

    return (hit
            .addBands(strata.multiply(hit).rename('stratum'))
            .addBands(t_break.multiply(hit).rename('t_break'))
            .addBands(confidence.multiply(hit).rename('confidence').toInt()))


def apply_mmu(clearing, min_mmu_ha=0.5, scale=10, max_size=1024):
    """Drop detections smaller than a minimum mapping unit.

    Applied after the stratum rules so the threshold acts on the final product,
    and eight-connected so diagonal slivers of a paddock edge stay joined.
    """
    clearing = ee.Image(clearing)
    if not min_mmu_ha:
        return clearing
    min_pixels = int(round(min_mmu_ha * 10000.0 / (scale * scale)))
    if min_pixels <= 1:
        return clearing
    hit = clearing.select('clearing')
    size = hit.selfMask().connectedPixelCount(maxSize=max_size,
                                              eightConnected=True)
    keep = hit.And(size.unmask(0).gte(min_pixels))
    return (keep.rename('clearing')
            .addBands(clearing.select('stratum').multiply(keep))
            .addBands(clearing.select('t_break').multiply(keep))
            .addBands(clearing.select('confidence').multiply(keep)))
