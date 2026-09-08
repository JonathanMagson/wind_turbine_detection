"""NSW area-of-interest presets."""

import ee

# Bounding boxes as (west, south, east, north) in EPSG:4326.
# Chosen in NSW landscapes where SLATS has repeatedly reported woody clearing,
# spanning the forest / woodland / grassland gradient across the state.
AOIS = {
    # Brigalow Belt South / north-west slopes cropping frontier.
    'moree': (149.60, -29.60, 150.00, -29.30),
    'walgett': (148.00, -30.20, 148.40, -29.90),
    'collarenebri': (148.40, -29.70, 148.80, -29.40),
    'narrabri': (149.55, -30.45, 149.95, -30.15),
    # Pilliga / Nandewar woodland margin.
    'pilliga': (148.90, -30.90, 149.30, -30.60),
    # Riverina mallee and box woodland edge, grassland dominated.
    'hay': (144.60, -34.70, 145.00, -34.40),
    # Western plains grassland / chenopod shrubland.
    'cobar': (145.60, -31.60, 146.00, -31.30),
    # South-east forestry and clearing mix.
    'bombala': (149.00, -37.00, 149.40, -36.70),
}


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
