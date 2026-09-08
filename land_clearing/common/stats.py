"""Area reductions shared by both pipelines."""

import ee

from . import masks


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


def area_by_stratum(clearing, aoi, scale=10, max_pixels=1e13):
    """Detected cleared hectares per cover stratum (client-side dict by name)."""
    stratum = ee.Image(clearing).select('stratum')
    stats = (ee.Image.pixelArea().addBands(stratum)
             .reduceRegion(reducer=ee.Reducer.sum().group(groupField=1,
                                                          groupName='stratum'),
                           geometry=aoi, scale=scale, maxPixels=max_pixels))
    groups = ee.List(stats.get('groups')).getInfo() or []
    out = {name: 0.0 for name in masks.STRATUM_NAMES.values()}
    for g in groups:
        code = int(g['stratum'])
        if code in masks.STRATUM_NAMES:
            out[masks.STRATUM_NAMES[code]] = g['sum'] / 10000.0
    return out


def area_by_year(clearing, aoi, start_year, end_year, scale=10,
                 max_pixels=1e13):
    """Detected cleared hectares per calendar year, from a ``t_break`` band.

    ``t_break`` is a fractional year (CCDC ``dateFormat=1``), so the calendar
    year is just its floor.
    """
    year = ee.Image(clearing).select('t_break').floor().toInt()
    stats = (ee.Image.pixelArea().addBands(year)
             .reduceRegion(reducer=ee.Reducer.sum().group(groupField=1,
                                                          groupName='year'),
                           geometry=aoi, scale=scale, maxPixels=max_pixels))
    groups = ee.List(stats.get('groups')).getInfo() or []
    out = {y: 0.0 for y in range(int(start_year), int(end_year) + 1)}
    for g in groups:
        y = int(g['year'])
        if y in out:
            out[y] = g['sum'] / 10000.0
    return out
