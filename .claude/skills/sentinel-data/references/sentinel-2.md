# Sentinel-2 (optical)

## The bucket route — no credentials

`sentinel-cogs` (Element 84, us-west-2), Level-2A surface reflectance as
Cloud-Optimized GeoTIFFs.

```
https://sentinel-cogs.s3.us-west-2.amazonaws.com/
  sentinel-s2-l2a-cogs/{utm_zone}/{lat_band}/{square}/{year}/{month}/{SCENE}/
      B01.tif … B12.tif        the bands; 10 m ones are 10980 x 10980 uint16
      SCL.tif                  20 m scene classification — screen with this
      TCI.tif                  true-colour composite, handy for eyeballing
      AOT.tif  WVP.tif         aerosol optical thickness, water vapour
      {SCENE}.json             STAC item: datetime, eo:cloud_cover, sun angles
      granule_metadata.xml     per-band view angles, exact sensing time
      thumbnail.jpg
```

Month has no leading zero (`/2024/6/`, not `/2024/06/`). Scene ids look like
`S2B_55HGB_20240612_0_L2A`.

List with the plain S3 REST API — no SDK, no credentials:

```bash
B=https://sentinel-cogs.s3.us-west-2.amazonaws.com
curl -s "$B/?list-type=2&delimiter=/&prefix=sentinel-s2-l2a-cogs/55/H/GB/2024/6/&max-keys=200"
```

## Bands

| Asset | Band | GSD | Use |
|---|---|---|---|
| B02 B03 B04 | blue, green, red | 10 m | true colour, most indices |
| B08 | NIR | 10 m | NDVI with B04 |
| B05 B06 B07 B8A | red edge / narrow NIR | 20 m | vegetation, chlorophyll |
| B11 B12 | SWIR | 20 m | moisture, burn indices, geology |
| B01 B09 | aerosol, water vapour | 60 m | atmospheric, rarely analysed directly |
| SCL | classification | 20 m | cloud/shadow/water masking |

L2A values are reflectance × 10000. Since processing baseline 04.00 they also
carry a BOA offset; the STAC item's `earthsearch:boa_offset_applied` says
whether the COG already has it applied. Differences between pixels are
unaffected either way — it matters for absolute reflectance, not for contrast.

## SCL classes

`0` no data · `1` saturated/defective · `2` dark area · `3` cloud shadow ·
`4` vegetation · `5` bare soil · `6` water · `7` unclassified ·
`8` cloud medium probability · `9` cloud high probability · `10` thin cirrus ·
`11` snow/ice

Screen a window on `(3, 8, 9, 10)` for cloud and `0` for no data. Include `11`
where snow would confound the analysis.

## STAC route — when the API is reachable

Much less bookkeeping, because you search by geometry instead of resolving MGRS
tiles yourself.

```python
from pystac_client import Client
cat = Client.open("https://earth-search.aws.element84.com/v1")
items = cat.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime="2024-05-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 20}},
).item_collection()
href = items[0].assets["blue"].href     # same COGs as the bucket route
```

Asset keys are names, not band ids: `blue`, `green`, `red`, `nir`, `swir16`,
`swir22`, `scl`, `visual`. Still screen the window on SCL — `eo:cloud_cover` is
a whole-tile number.

Useful item properties: `datetime`, `eo:cloud_cover`, `proj:epsg`,
`grid:code` (MGRS), `view:sun_azimuth`, `view:sun_elevation`,
`s2:nodata_pixel_percentage`.

## Viewing and sun angles

`granule_metadata.xml` in each scene folder holds the authoritative geometry:

- `SENSING_TIME` — exact acquisition instant.
- `Mean_Sun_Angle` — `ZENITH_ANGLE`, `AZIMUTH_ANGLE` (elevation = 90 − zenith).
- `Mean_Viewing_Incidence_Angle` per `bandId` — zenith and azimuth per band.

`bandId` order is `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12`
(0–12). Off-nadir zenith is typically 2–5°, so the parallax shift of a tall
object is small but not zero: a 100 m tower at 4° leans about 7 m.

## L1C, other baselines, reprocessed archives

Not in `sentinel-cogs` — that bucket is L2A only. Use the Copernicus Data Space
Ecosystem (free registration; OData / OpenSearch / STAC plus S3 with generated
credentials) when you need top-of-atmosphere radiance, a specific processing
baseline, or a reprocessed collection.

## Australia: Digital Earth Australia

`dea-public-data` (ap-southeast-2, open) carries Geoscience Australia's ARD,
already terrain- and BRDF-corrected, in Australian projections:

```
baseline/ga_s2am_ard_3/  ga_s2bm_ard_3/  ga_s2cm_ard_3/   Sentinel-2 ARD (2A, 2B, 2C)
baseline/ga_ls8c_ard_3/  ga_ls9c_ard_3/  …                Landsat ARD
derivative/ga_ls_wo_3/  ga_ls_fc_3/  …                    water observations, fractional cover
```

Sentinel-2 and Landsat only — there is no Sentinel-1 collection here, so SAR
still comes from the routes in `sentinel-1.md`. Prefer DEA over raw L2A for
Australian work where consistency across dates matters more than having the
original pixels; prefer `sentinel-cogs` when you want the unmodified ESA
product or need to match a specific acquisition.
