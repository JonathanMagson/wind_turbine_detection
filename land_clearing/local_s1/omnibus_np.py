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
