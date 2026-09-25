---
name: sentinel-data
description: Find and download Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery and cut analysis-ready windows out of it. Use this whenever a task needs Copernicus or Sentinel imagery, satellite bands over an area of interest, cloud-free scene selection, NDVI or other band maths, change detection over time, SAR backscatter, or any "get me imagery for this lat/lon" request — including when the user names a place rather than coordinates, or wants analysis-ready data over Australia. Reach for it before writing any code that calls a STAC API, opens a COG over HTTP, or downloads a satellite scene — it covers which endpoints an agent session can actually reach, how to read only the window you need instead of a 240 MB band, and the metadata traps (tile-level cloud cover, MGRS band edges, UTM hemisphere) that silently produce wrong answers rather than errors.
---

# Sentinel-1 and Sentinel-2 imagery

Getting the pixels is rarely the hard part. What goes wrong is choosing a scene
that turns out to be cloud over your target, downloading a gigabyte to use four
megabytes of it, or accepting a metadata field that quietly describes a
different place. This skill front-loads those.

## 1. Probe before you plan

Do this first, every time, before designing anything. Agent sessions differ:
Claude Code on the web sits behind an egress allowlist where public S3 buckets
usually work and catalogue **APIs** usually do not; a local CLI has your normal
network and the APIs are the better route. Finding out after you've written a
`pystac_client` pipeline wastes the whole design.

```bash
probe () { printf "%-64s " "$1"; curl -s -m 25 -o /dev/null -w "%{http_code}\n" "$1" || echo blocked; }
probe "https://sentinel-cogs.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=1"          # S2 L2A COGs
probe "https://sentinel-s1-l1c.s3.amazonaws.com/?list-type=2&max-keys=1"                  # S1 GRD
probe "https://dea-public-data.s3.ap-southeast-2.amazonaws.com/?list-type=2&max-keys=1"   # Australian ARD
probe "https://earth-search.aws.element84.com/v1/"                                        # STAC
probe "https://planetarycomputer.microsoft.com/api/stac/v1/"                              # STAC
probe "https://catalogue.dataspace.copernicus.eu/stac/"                                   # Copernicus
```

`200` means usable. `000` is the proxy refusing the tunnel — not a transient
error, don't retry it. Tell the user which route you're taking and why; "the
STAC API is blocked here so I'm walking the bucket by tile" is useful
information, silently producing worse results is not.

Install what the chosen route needs:

```bash
pip install rasterio numpy mgrs                        # bucket route
pip install pystac-client planetary-computer asf_search  # API routes
```

`rasterio` bundles GDAL and picks up `https_proxy` by itself — proxied sessions
need no extra configuration.

## 2. Pick the source

| You need | Go to | Auth |
|---|---|---|
| Optical, surface reflectance, anywhere | `sentinel-cogs` bucket (L2A COGs) | none |
| Optical, with spatial search | earth-search STAC → same COGs | none |
| Optical, L1C / specific baseline / reprocessed | Copernicus Data Space | free account |
| Optical or Landsat ARD over Australia | `dea-public-data` bucket | none |
| SAR, willing to preprocess | `sentinel-s1-l1c` bucket (GRD) | none |
| SAR, standard route, whole products | ASF (`asf_search`) | free Earthdata login |
| SAR, analysis-ready backscatter | Planetary Computer `sentinel-1-rtc` | free key for some collections |

Then read the matching reference file — don't work from memory, the path
layouts and field names are fiddly:

- **`references/sentinel-2.md`** — bucket layout, band table, the STAC snippet,
  SCL classes, per-band viewing angles, and the DEA collections.
- **`references/sentinel-1.md`** — the three routes, what each still owes you in
  preprocessing, and why a GRD file has no CRS.

## 3. Read windows, not scenes

A Sentinel-2 10 m band is 10980 × 10980 — about 240 MB. These are
Cloud-Optimized GeoTIFFs, so HTTP range requests fetch only the tiles you touch;
a 512² window takes a few seconds. Reading the whole band to crop 1 % of it is
the most common waste in this work.

```python
import rasterio, utm
from rasterio.windows import Window

with rasterio.open(url) as src:
    east, north, *_ = utm.from_latlon(lat, lon)
    row, col = src.index(east, north)
    if not (0 <= row < src.height and 0 <= col < src.width):
        raise ValueError("point is outside this tile — see the MGRS note below")
    side = 1400                                   # 14 km at 10 m
    win = Window(col - side // 2, row - side // 2, side, side)
    data = src.read(1, window=win)
    transform = src.window_transform(win)         # keep it, or you lose georeferencing
```

Always carry `window_transform` into whatever you write out. A cropped array
with the parent transform is georeferenced to the wrong place, and nothing will
tell you.

## 4. Choose the scene by your window, not by the tile

This is the highest-value habit in the skill. Tile-level statistics describe
~12 000 km². A 14 km window can be solid cloud inside a tile reporting 0.3 %
cloud, and a tile reporting 18 % no-data can cover your target completely.
Screening on `eo:cloud_cover` alone produces confidently wrong scene choices.

Read the same footprint out of the 20 m `SCL.tif` classification band and count
the classes you care about — 3 (cloud shadow), 8 and 9 (cloud), 10 (cirrus),
0 (no data):

```python
scl, *_ = cut_window(scene, "SCL", lat, lon, km)     # same geographic footprint
cloudy = np.isin(scl, (3, 8, 9, 10)).mean()
empty  = (scl == 0).mean()
```

Rank the survivors by what the task actually needs. For anything shadow-based —
object height, terrain, detecting structures by the shadow they cast — that
means **lowest sun elevation**, not lowest cloud: a 25° winter sun throws a
170 m shadow off an 80 m tower where a 65° summer sun throws 37 m. For
reflectance, NDVI or classification work, prefer clearest and closest to the
date of interest, and note that low sun brings long shadows and BRDF effects
you may not want. Say which you optimised for.

## 5. Resolving the MGRS tile

Sentinel-2 files are organised by MGRS tile (`55/H/GB/...`). Converting a
lat/lon to a tile is not quite deterministic: a point within a few kilometres of
a latitude-band edge converts to a square the archive files under the
*neighbouring* band. A real example — 31.93°S converts to `54JWK`, while the
imagery lives under `54HWK`.

So treat the conversion as a first guess: walk the ring of neighbouring tiles
(vary both square letters and the band letter) and keep the first whose raster
actually contains your point. Verify with `src.bounds`, not by assumption — the
failure mode otherwise is a window silently clipped to the edge of the wrong
tile, which looks like perfectly good data.

`scripts/fetch_s2_window.py` implements all of this. Prefer running or adapting
it over rewriting it:

```bash
python scripts/fetch_s2_window.py --lat -34.5753 --lon 148.8701 --km 14 \
    --year 2024 --months 5,6,7,8 --band B02 --name ryepark --out ./scenes
```

It resolves the tile with the band-edge fallback, lists candidates, screens each
on its SCL window, picks by sun elevation, cuts the window, and writes a GeoTIFF
plus a JSON sidecar carrying the acquisition time, sun and per-band satellite
view angles (pulled from the scene's `granule_metadata.xml`, which is more
reliable than any summary field). `--sidecar gdalinfo` additionally emits a
`gdalinfo`-style text sidecar for tools that parse that format.

## 6. Traps that produce wrong answers rather than errors

- **UTM hemisphere.** In `WGS 84 / UTM zone 55S` the `S` is the hemisphere — but
  it is also a valid *northern* MGRS band letter. Passing it to `utm.to_latlon`
  as a band letter puts a southern-hemisphere scene thousands of kilometres
  north, and anything derived from position (sun angles, distances) follows it.
  Parse the hemisphere from the projection, and pass `northern=` explicitly.
- **Copied metadata.** Sidecars assembled by hand or by templating often carry
  fields from the scene they were copied from. If a per-band angle table is
  identical across two scenes years and continents apart, it is stale. Prefer
  the scene's own `granule_metadata.xml`; cross-check an `epsg=` tag against the
  projection block rather than trusting it.
- **Disagreeing timestamps.** Acquisition time appears in several fields that
  can differ by hours. If a derived sun elevation comes out near or below zero,
  or the sun is in an implausible part of the sky for the local time, you have
  the wrong field — check it against the other candidates rather than carrying
  on.
- **Sanity-check position early.** Compute the scene centre's lat/lon and say it
  out loud in your first message. Most of the traps above surface immediately as
  "that's not where I asked for".

## 7. Report what you got

When handing imagery back, state the scene id, acquisition datetime, tile,
cloud fraction *in the window*, sun elevation and azimuth, pixel size and the
window's geographic bounds. These are what make a result reproducible, and they
are exactly what someone needs to judge whether the scene suits their question.
