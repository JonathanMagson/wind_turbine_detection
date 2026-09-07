"""NumPy port of the sequential omnibus change detection test.

Same algorithm as ``land_clearing.omnibus_s1.omnibus``, which runs on Earth
Engine; this one runs locally on numpy arrays so the pipeline needs no Earth
Engine account. Ported from the Earth Engine Community tutorial "Detecting
Changes in Sentinel-1 Imagery" parts 2-3 (Apache-2.0), which implements:

  K. Conradsen, A. A. Nielsen and H. Skriver (2016), "Determining the Points of
  Change in Time Series of Polarimetric SAR Data", IEEE TGRS 54(5), 3007-3024.

Input is a list of dual-pol images shaped ``(2, H, W)`` holding *linear power*
sigma-nought for [VV, VH], each multiplied by the equivalent number of looks.
Only the diagonal of the covariance matrix is used, so the determinant is the
product of the two channels.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import chi2

# Equivalent number of looks for Sentinel-1 IW GRDH.
ENL = 4.4

# bmap codes.
NO_CHANGE, POSITIVE, NEGATIVE, INDEFINITE = 0, 1, 2, 3


def _det(im):
    """Determinant of the 2x2 diagonal covariance matrix."""
    return im[0] * im[1]


def _log_det_sum(im_list, j):
    """log determinant of the sum of the first j images."""
    return np.log(_det(np.sum(im_list[:j], axis=0)))


def _log_det(im_list, j):
    """log determinant of the jth image (1-based)."""
    return np.log(_det(im_list[j - 1]))


def _pval(im_list, j, m=ENL):
    """Returns (P value, -2m logRj) for the jth image in the series."""
    j = float(j)
    m2logRj = (_log_det_sum(im_list, int(j) - 1) * (j - 1)
               + _log_det(im_list, int(j))
               + 2.0 * j * np.log(j)
               - 2.0 * (j - 1) * np.log(j - 1)
               - _log_det_sum(im_list, int(j)) * j) * (-2.0 * m)

    # Correction to the simple Wilks approximation.
    rhoj = 1.0 - (1.0 + 1.0 / (j * (j - 1))) / (6.0 * m)
    omega2j = -((1.0 - 1.0 / rhoj) ** 2) / 2.0
    x = m2logRj * rhoj
    c2, c6 = chi2.cdf(x, 2), chi2.cdf(x, 6)
    pv = 1.0 - (c2 + omega2j * c6 - omega2j * c2)
    return pv, m2logRj


def p_values(im_list, m=ENL):
    """P value array: rows ell = k..2, each holding pv for j=2..ell then pvQl."""
    k = len(im_list)
    pv_arr = []
    for ell in range(k, 1, -1):
        sub = im_list[k - ell:k]
        pvs, m2logQl = [], 0.0
        for j in range(2, ell + 1):
            pv1, m2logRj1 = _pval(sub, j, m)
            pvs.append(pv1)
            m2logQl = m2logQl + m2logRj1

        f = 2.0 * (ell - 1)
        rho = 1.0 - (ell / m - 1.0 / (ell * m)) / (3.0 * f)
        omega2 = -f * ((1.0 - 1.0 / rho) ** 2) / 4.0
        x = m2logQl * rho
        cf, cf4 = chi2.cdf(x, f), chi2.cdf(x, f + 4)
        pvs.append(1.0 - (cf + omega2 * cf4 - omega2 * cf))
        pv_arr.append(pvs)
    return pv_arr


def change_maps(im_list, median=False, alpha=0.01, m=ENL):
    """Thematic change maps for a dual-pol series.

    Returns a dict of:
      ``cmap`` interval index of the most recent significant change
      ``smap`` interval index of the first significant change
      ``fmap`` number of significant changes over the series
      ``bmap`` (k-1, H, W) per-interval codes: 0 none, 1 positive definite
               (backscatter rose), 2 negative definite (fell), 3 indefinite
    """
    im_list = [np.asarray(im, dtype=np.float64) for im in im_list]
    k = len(im_list)
    if k < 3:
        raise ValueError('Need at least 3 images, got %d.' % k)
    shape = im_list[0].shape[1:]

    pv_arr = p_values(im_list, m)

    cmap = np.zeros(shape)
    smap = np.zeros(shape)
    fmap = np.zeros(shape)
    bmap = np.zeros((k - 1,) + shape)

    for i, row in enumerate(pv_arr, start=1):
        pvs, pvQ = row[:-1], row[-1]
        if median:
            pvQ = median_filter(pvQ, size=5)
        for j, pv in enumerate(pvs, start=1):
            cmapj = i + j - 1
            # Rj significant, Ql significant, and still on row i.
            tst = (pv < alpha) & (pvQ < alpha) & (cmap == i - 1)
            cmap[tst] = cmapj
            fmap[tst] += 1
            if i == 1:
                smap[tst] = cmapj
            bmap[i + j - 2][tst] = 1

    # Second pass: classify each flagged change by direction.
    avimg = im_list[0].copy()
    count = np.ones(shape)
    for j, image in enumerate(im_list[1:]):
        diff = image - avimg
        det_diff = _det(diff)
        posd = (diff[0] > 0) & (det_diff > 0)
        negd = (diff[0] < 0) & (det_diff > 0)

        changed = bmap[j] > 0
        bmap[j][changed] = INDEFINITE
        bmap[j][changed & posd] = POSITIVE
        bmap[j][changed & negd] = NEGATIVE

        # Provisional means, reset wherever a change was detected.
        count += 1
        avimg = avimg + (image - avimg) / count
        avimg[:, changed] = image[:, changed]
        count[changed] = 1

    return {'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap}


# Detection limit, measured over sparse mulga near Cobar against a clearing
# event confirmed by its persistent 3.5 dB divergence from surrounding
# woodland (33 ha, September-October 2023). Recall of that event, and the
# false-positive rate over known-stable woody vegetation, both at alpha=0.01:
#
#   multilook  pixel   ENL    recall   false positive
#           1    11 m  4.40    26.0%            0.24%
#           3    33 m  6.17    30.4%            0.11%
#           5    56 m  8.78    57.5%            0.22%
#           8    89 m 11.35    83.3%            0.18%
#
# Recall triples while the false-positive rate falls, which is what a purely
# speckle-limited problem looks like: the extra looks buy sensitivity at no
# cost in specificity. Closed-canopy clearing gives a much larger step and
# needs far less multi-looking; 8 is calibrated for the sparse arid case.


def multilook(image, factor):
    """Block-average a (bands, H, W) image by ``factor`` in linear power.

    Averaging must happen in linear power, not dB. Non-overlapping blocks are
    used rather than a sliding boxcar, because overlapping windows leave
    neighbouring output pixels correlated, which would violate the test's
    independence assumption while appearing to add looks.
    """
    if factor <= 1:
        return image
    b, h, w = image.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    trimmed = image[:, :h2, :w2]
    return trimmed.reshape(b, h2 // factor, factor,
                           w2 // factor, factor).mean(axis=(2, 4))


def estimate_enl(stack, mask=None, window=7, min_samples=500):
    """Estimate the equivalent number of looks from the imagery itself.

    For single-channel intensity (linear power) speckle, ENL = (mean/std)^2.
    Computed on local windows and reduced by the median, because scene texture
    only ever *adds* variance and so biases the estimate downward -- which is
    the safe direction: understating ENL makes the change test stricter, not
    more permissive.

    Assuming a nominal ENL after multi-looking would be wrong here: Sentinel-1
    GRD is posted at 10 m from roughly 20 m resolution, so neighbouring pixels
    are correlated and an NxN block does not deliver N^2 independent looks.
    """
    from scipy.ndimage import uniform_filter

    ratios = []
    for image in stack:
        for band in np.asarray(image, dtype=np.float64):
            valid = np.isfinite(band) & (band > 0)
            if mask is not None:
                valid = valid & mask
            if valid.sum() < min_samples:
                continue
            filled = np.where(valid, band, np.nan)
            filled = np.nan_to_num(filled, nan=np.nanmedian(band[valid]))
            mean = uniform_filter(filled, window)
            mean_sq = uniform_filter(filled ** 2, window)
            var = np.clip(mean_sq - mean ** 2, 1e-20, None)
            enl_local = (mean ** 2) / var
            # Drop window edge effects and non-finite values.
            edge = window // 2 + 1
            core = enl_local[edge:-edge, edge:-edge]
            core_valid = valid[edge:-edge, edge:-edge]
            sel = core[core_valid & np.isfinite(core)]
            if sel.size:
                ratios.append(np.median(sel))
    if not ratios:
        raise ValueError('Could not estimate ENL: no valid homogeneous samples.')
    return float(np.median(ratios))


# Calibrated equivalent number of looks per multi-look factor, measured in the
# controlled experiment documented above. Used in preference to estimating per
# AOI: the multi-look gain comes from the *sensor's* correlation structure, not
# from the scene, whereas the local-window estimator is strongly scene
# dependent -- it inflates over homogeneous ground (less texture to add
# variance) and so made the test far too permissive on the most uniform AOIs,
# where estimates ran from 12 to 30 for the same factor.
CALIBRATED_ENL = {1: 4.40, 3: 6.17, 5: 8.78, 8: 11.35}


def calibrated_enl(factor):
    """Equivalent number of looks for a multi-look factor, log-interpolated."""
    factor = max(1, int(factor))
    if factor in CALIBRATED_ENL:
        return CALIBRATED_ENL[factor]
    xs = sorted(CALIBRATED_ENL)
    ys = [CALIBRATED_ENL[x] for x in xs]
    return float(np.interp(np.log(factor), np.log(xs), ys))


def normalise_common_mode(stack, mask=None):
    """Remove date-to-date global radiometric shifts. Returns (stack, factors).

    A rain front changes backscatter across a whole scene at once. That is a
    real change and the omnibus test flags it correctly -- over an entire AOI,
    which is useless for clearing. In one Cobar run a single wet-to-dry
    transition produced 20,454 ha of "clearing".

    Each date and polarisation is rescaled so its median over ``mask``
    (normally the woody baseline) matches the series geometric mean. Localised
    change survives because the median is robust: clearing is a small fraction
    of any sensible AOI, so it barely moves the reference level, while a
    basin-wide moisture shift moves it almost entirely.
    """
    stack = [np.asarray(im, dtype=np.float64) for im in stack]
    n_bands = stack[0].shape[0]
    levels = np.empty((len(stack), n_bands))
    for t, im in enumerate(stack):
        for b in range(n_bands):
            band = im[b]
            sel = np.isfinite(band) & (band > 0)
            if mask is not None:
                sel = sel & mask
            levels[t, b] = np.median(band[sel]) if sel.any() else np.nan

    target = np.exp(np.nanmean(np.log(levels), axis=0))
    out, factors = [], []
    for t, im in enumerate(stack):
        f = np.where(np.isfinite(levels[t]) & (levels[t] > 0),
                     target / levels[t], 1.0)
        out.append(im * f[:, None, None])
        factors.append(f)
    return out, np.array(factors)
