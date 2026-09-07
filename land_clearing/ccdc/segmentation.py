"""Run ``ee.Algorithms.TemporalSegmentation.Ccdc`` and extract usable breaks.

CCDC returns per-pixel *array* bands, one element per fitted segment: ``tStart``,
``tEnd``, ``tBreak``, ``changeProb``, ``numObs``, plus ``<band>_coefs``,
``<band>_rmse`` and ``<band>_magnitude``. A pixel with no change has one
segment and no break; a pixel cleared once has two segments and one break.

``extract_breaks`` reduces that to flat image bands for a reporting window:
which break came first, when, how confident, and how big the jump was in each
band. Magnitude sign is what carries direction -- negative means the observed
value fell below what the seasonal model predicted.
"""

import ee

# ``dateFormat=1`` gives fractional years, which are directly comparable to a
# calendar window and readable in exported rasters.
DATE_FORMAT_FRACTIONAL_YEAR = 1


def run_ccdc(collection, breakpoint_bands, min_observations=4,
             chi_square_probability=0.99, min_years_scaler=1.33,
             lambda_=20.0 / 10000.0, max_iterations=25, tmask_bands=None):
    """Fit CCDC over ``collection``.

    ``min_observations`` is how many consecutive off-model observations are
    needed to confirm a break. The CCDC default of 6 was set for Landsat's
    16-day revisit; at Sentinel-1's 6-12 days that would demand two to three
    months of confirmation, so the default here is 4.

    ``tmask_bands`` is CCDC's own optical cloud screening. It is left off: the
    Sentinel-2 stack is already SCL-masked, and it is meaningless for radar.
    """
    # The server argument is literally named "lambda", which is a Python
    # keyword. The Earth Engine client matches keyword argument names against
    # the server signature exactly and rejects anything else, so it has to be
    # passed through a dict rather than as ``lambda_=``.
    kwargs = {
        'collection': ee.ImageCollection(collection),
        'breakpointBands': list(breakpoint_bands),
        'minObservations': min_observations,
        'chiSquareProbability': chi_square_probability,
        'minNumOfYearsScaler': min_years_scaler,
        'dateFormat': DATE_FORMAT_FRACTIONAL_YEAR,
        'lambda': lambda_,
        'maxIterations': max_iterations,
    }
    if tmask_bands:
        kwargs['tmaskBands'] = list(tmask_bands)
    return ee.Algorithms.TemporalSegmentation.Ccdc(**kwargs)


def year_fraction(date_str):
    """'YYYY-MM-DD' -> fractional year, matching ``dateFormat=1`` output."""
    d = ee.Date(date_str)
    year = d.get('year')
    start = ee.Date.fromYMD(year, 1, 1)
    next_start = ee.Date.fromYMD(ee.Number(year).add(1), 1, 1)
    elapsed = d.difference(start, 'day')
    length = next_start.difference(start, 'day')
    return ee.Number(year).add(elapsed.divide(length))


def extract_breaks(ccdc, bands, start_year, end_year, prefix=''):
    """Flatten CCDC arrays to the first break inside ``[start_year, end_year)``.

    Returns an image with bands (``prefix`` prepended to each):
      ``n_breaks``     number of breaks inside the window
      ``t_break``      fractional year of the first one, 0 if none
      ``change_prob``  CCDC change probability for that break
      ``<band>_mag``   magnitude of that break per input band, signed

    Empty arrays are handled by padding to length 1 and reading element 0, which
    yields 0 for pixels with no break rather than failing the reduction.
    """
    ccdc = ee.Image(ccdc)
    t_break = ccdc.select('tBreak')
    in_window = t_break.gte(ee.Number(start_year)).And(
        t_break.lt(ee.Number(end_year)))

    masked = t_break.arrayMask(in_window)
    n_breaks = masked.arrayLength(0).rename('n_breaks')
    first_break = masked.arrayPad([1]).arrayGet([0]).rename('t_break')

    change_prob = (ccdc.select('changeProb').arrayMask(in_window)
                   .arrayPad([1]).arrayGet([0]).rename('change_prob'))

    out = n_breaks.addBands(first_break).addBands(change_prob)
    names = ['n_breaks', 't_break', 'change_prob']
    for band in bands:
        mag = (ccdc.select('%s_magnitude' % band).arrayMask(in_window)
               .arrayPad([1]).arrayGet([0]).rename('%s_mag' % band))
        out = out.addBands(mag)
        names.append('%s_mag' % band)

    # No explicit zero-fill is needed: arrayPad pads an empty (no break in
    # window) array with 0, so arrayGet([0]) already returns 0 for those
    # pixels in every band above.
    if prefix:
        # Built locally rather than from bandNames().getInfo(), to keep this
        # function free of round trips.
        out = out.rename([prefix + n for n in names])
    return out
