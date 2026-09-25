# Sentinel-1 (SAR)

SAR is not a grey version of optical imagery. There are no shadows to measure in
the optical sense, speckle is everywhere, brightness means surface roughness and
geometry rather than colour, and the raw product is not in map coordinates at
all. Pick the route that matches how much of that you want to handle.

## Route A — analysis-ready, least work: Planetary Computer RTC

`sentinel-1-rtc` is radiometrically terrain-corrected gamma0: geocoded,
calibrated, ready to use. If you want backscatter over an area and don't want to
run a SAR pipeline, stop here.

```python
import planetary_computer as pc, pystac_client, rasterio
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)
items = cat.search(collections=["sentinel-1-rtc"],
                   intersects={"type": "Point", "coordinates": [lon, lat]},
                   datetime="2024-06-01/2024-06-30").item_collection()
with rasterio.open(items[0].assets["vv"].href) as src:   # signed href
    ...                                                  # windowed reads work
```

Some PC collections need a free API key; `sentinel-1-grd` is open. Requires the
API to be reachable — check the probe in SKILL.md.

## Route B — the standard archive: ASF

Alaska Satellite Facility, free NASA Earthdata login, the best search for GRD
and SLC.

```python
import asf_search as asf
results = asf.geo_search(
    platform=asf.PLATFORM.SENTINEL1,
    intersectsWith=f"POINT({lon} {lat})",
    start="2024-06-01", end="2024-06-30",
    processingLevel=asf.PRODUCT_TYPE.GRD_HD,
    beamMode=asf.BEAMMODE.IW,
)
session = asf.ASFSession().auth_with_creds(user, password)
results[0].download(path="./data", session=session)
```

Whole products, roughly a gigabyte each — no windowed reads. Take credentials
from the environment, never inline, and never echo them.

## Route C — straight from the bucket, no credentials

`sentinel-s1-l1c`, GRD products:

```
https://sentinel-s1-l1c.s3.amazonaws.com/
  GRD/{year}/{month}/{day}/{mode}/{pol}/{PRODUCT_ID}/
      measurement/iw-vv.tiff        (or ew-hh.tiff, iw-vh.tiff, …)
      annotation/                   calibration, noise, orbit state vectors
```

The measurement files are tiled 1024² with overviews, so windowed and
overview-level reads work well. **But the file has no CRS and no affine
transform** — it is in radar geometry, georeferenced by a few hundred GCPs in
EPSG:4326. A representative product: 10476 × 10708, uint16, 483 GCPs.

Nothing spatial works until you warp:

```bash
gdalwarp -tps -t_srs EPSG:32755 -tr 20 20 \
  /vsicurl/https://sentinel-s1-l1c.s3.amazonaws.com/GRD/.../measurement/iw-vv.tiff \
  out.tif
```

That yields geocoded but *uncalibrated* DN. Real work then needs radiometric
calibration from the annotation XML, speckle filtering, and ideally terrain
correction with a DEM. This bucket is documented as requester-pays for S3 API
access; anonymous HTTPS GETs have worked in practice, and if yours is refused,
use credentials with `--request-payer requester`.

## Route D — Copernicus Data Space

Free registration, every level including SLC, plus OData / OpenSearch / STAC
and S3 access. Use when you need SLC (interferometry, coherence) or a specific
reprocessed collection.

## Choosing modes and polarisations

- **IW** (Interferometric Wide) — the default over land, 250 km swath, ~10 m
  pixel spacing in GRD-HD. Use unless you have a reason not to.
- **EW** (Extra Wide) — polar and maritime, coarser (~40 m).
- **VV+VH** over land, **HH+HV** over ice and some maritime. VV for general
  backscatter; the cross-pol channel (VH/HV) carries volume scattering and is
  often the more useful of the two for vegetation and structure.

## Pitfalls

- **Comparing dates means comparing geometry.** Backscatter depends strongly on
  incidence angle and look direction, so only compare scenes from the same
  relative orbit and pass direction unless you are using terrain-corrected
  gamma0 built for cross-orbit comparison.
- **Speckle is multiplicative.** Filter (Lee, refined Lee, or multi-look) before
  thresholding or differencing; don't treat single pixels as measurements.
- **dB or linear.** Backscatter is usually shown in dB (`10 * log10(gamma0)`),
  but statistics should be computed in linear power and converted afterwards —
  averaging dB values is not the same as the dB of the average.
