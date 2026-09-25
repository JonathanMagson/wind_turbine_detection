# Getting Sentinel-1 and Sentinel-2 data inside Claude Code

Notes from building `demo/fetch_scene.py`. Everything marked **verified** was
actually run from a Claude Code session on the web on 9 Sep 2026; the rest is
signposting. Useful beyond this repository — most of it is about which endpoints
an agent session can actually reach, and which metadata fields lie.

> A reusable version of this, which Claude Code loads automatically, lives in
> [`.claude/skills/sentinel-data/`](../.claude/skills/sentinel-data/SKILL.md)
> — same material, plus a working fetch script and per-mission reference files.

---

## 0. First, find out what your session can reach

This matters more than anything else below. There are two very different
environments:

- **Claude Code on the web / cloud sessions** run behind an egress proxy with an
  allowlist. Most public S3 buckets work. Most catalogue *APIs* do not.
- **Claude Code CLI on your own machine** has your normal network. Everything
  below works, and the STAC APIs are the better route.

Have Claude run this before planning anything:

```bash
probe () { printf "%-64s " "$1"; curl -s -m 25 -o /dev/null -w "%{http_code}\n" "$1" || echo blocked; }
probe "https://sentinel-cogs.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=1"      # S2 L2A COGs
probe "https://sentinel-s1-l1c.s3.amazonaws.com/?list-type=2&max-keys=1"              # S1 GRD
probe "https://dea-public-data.s3.ap-southeast-2.amazonaws.com/?list-type=2&max-keys=1"  # DEA (Australia)
probe "https://earth-search.aws.element84.com/v1/"                                    # STAC search
probe "https://planetarycomputer.microsoft.com/api/stac/v1/"                          # STAC search
probe "https://catalogue.dataspace.copernicus.eu/stac/"                               # Copernicus
probe "https://api.daac.asf.alaska.edu/services/search/param?platform=S1&maxResults=1&output=json"
```

What I got from a web session (**verified**): the three buckets returned `200`;
every API returned `000` (proxy refused the CONNECT tunnel). So on the web you
navigate buckets by path; locally you use STAC search. Plan accordingly — a
blocked STAC API is the single thing most likely to derail this.

Install once:

```bash
python3 -m venv .venv
.venv/bin/pip install rasterio numpy mgrs            # bucket route
.venv/bin/pip install pystac-client planetary-computer asf_search   # API routes
```

`rasterio` bundles GDAL and picks up `https_proxy` on its own — no extra config
needed in a proxied session (**verified**).

---

## 1. Sentinel-2 — the easy one, no credentials at all

Bucket `sentinel-cogs` (Element 84, us-west-2), Level-2A surface reflectance as
Cloud-Optimized GeoTIFFs. Path layout (**verified**):

```
sentinel-s2-l2a-cogs/{utm_zone}/{lat_band}/{square}/{year}/{month}/{SCENE}/
    B01.tif … B12.tif      10/20/60 m bands, 10980 x 10980 for the 10 m ones
    SCL.tif                20 m scene classification (cloud, shadow, water …)
    TCI.tif                true-colour composite
    {SCENE}.json           STAC item: datetime, eo:cloud_cover, sun angles
    granule_metadata.xml   per-band satellite view angles, exact sensing time
```

e.g. `sentinel-s2-l2a-cogs/55/H/GB/2024/6/S2B_55HGB_20240612_0_L2A/B02.tif`

**Read only the window you need.** These are COGs, so range requests work — a
512 x 512 window out of a 10980² file took ~3 s (**verified**):

```python
import rasterio, utm
from rasterio.windows import Window

url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/55/H/GB/2024/6/S2B_55HGB_20240612_0_L2A/B02.tif"
with rasterio.open(url) as src:
    east, north, *_ = utm.from_latlon(-35.09, 149.51)
    row, col = src.index(east, north)
    side = 1400                                  # 14 km at 10 m
    win = Window(col - side // 2, row - side // 2, side, side)
    data = src.read(1, window=win)
    transform = src.window_transform(win)        # keep this to stay georeferenced
```

Never `src.read(1)` the whole thing unless you mean it — that's ~240 MB per band.

**With a STAC API (local sessions)** it's less bookkeeping, because you can
search by point instead of resolving MGRS tiles yourself:

```python
from pystac_client import Client
cat = Client.open("https://earth-search.aws.element84.com/v1")
items = cat.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [149.51, -35.09]},
    datetime="2024-05-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 20}},
).item_collection()
href = items[0].assets["blue"].href          # then the same windowed read
```

L2A only in that bucket. For L1C top-of-atmosphere, use Copernicus (§4).

---

## 2. Sentinel-1 — three routes, pick by how much preprocessing you want

**(a) Straight from the bucket — no credentials** (**verified**). Bucket
`sentinel-s1-l1c`, GRD products:

```
GRD/{year}/{month}/{day}/{mode}/{pol}/{PRODUCT_ID}/
    measurement/iw-vv.tiff       (or ew-hh.tiff etc.)
    annotation/…                 calibration, noise, orbit metadata
```

The measurement files are tiled 1024² with overviews down to 1/64, so windowed
and overview reads work well. **But**: `crs` is `None` and there is no affine
transform — the file is in radar geometry with a few hundred GCPs
(EPSG:4326). Verified on a real product: 10476 x 10708, uint16, 483 GCPs. You
must warp before anything spatial:

```bash
gdalwarp -tps -t_srs EPSG:32755 -tr 20 20 \
  /vsicurl/https://sentinel-s1-l1c.s3.amazonaws.com/GRD/.../measurement/iw-vv.tiff out.tif
```

That gets you geocoded *uncalibrated DN*. Real work also needs radiometric
calibration (the annotation XML), speckle filtering, and ideally terrain
correction. Note this bucket is documented as requester-pays for the S3 API;
anonymous HTTPS GETs worked in my test, so if yours 403s, use credentials and
`--request-payer requester`.

**(b) ASF — the standard route for GRD/SLC.** Free NASA Earthdata login,
excellent search:

```python
import asf_search as asf
results = asf.geo_search(
    platform=asf.PLATFORM.SENTINEL1,
    intersectsWith="POINT(149.51 -35.09)",
    start="2024-06-01", end="2024-06-30",
    processingLevel=asf.PRODUCT_TYPE.GRD_HD,
    beamMode=asf.BEAMMODE.IW,
)
session = asf.ASFSession().auth_with_creds("earthdata_user", "earthdata_pass")
results[0].download(path="./data", session=session)
```

Products are whole ~1 GB zips — no windowed reads. Store the credentials as
environment variables, not in the script, and never paste them into chat.

**(c) Planetary Computer — analysis-ready, least work.** `sentinel-1-rtc` is
radiometrically terrain-corrected gamma0: geocoded, calibrated, ready to use.
Some PC collections need a free API key; `sentinel-1-grd` is open.

```python
import planetary_computer as pc, pystac_client, rasterio
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)
items = cat.search(collections=["sentinel-1-rtc"],
                   intersects={"type": "Point", "coordinates": [149.51, -35.09]},
                   datetime="2024-06-01/2024-06-30").item_collection()
with rasterio.open(items[0].assets["vv"].href) as src:
    ...                     # signed href, windowed reads work
```

If you only want backscatter over an area and don't want to run a SAR pipeline,
use this one.

---

## 3. If the area of interest is Australia: DEA

`dea-public-data` (ap-southeast-2) is open and reachable (**verified**), with
Geoscience Australia's ARD — already terrain- and BRDF-corrected, in Australian
projections:

```
baseline/ga_s2am_ard_3/  ga_s2bm_ard_3/  ga_s2cm_ard_3/     Sentinel-2 ARD
baseline/ga_ls8c_ard_3/  ga_ls9c_ard_3/  …                  Landsat ARD
derivative/…                                                water observations, fractional cover, land cover
```

Sentinel-2 and Landsat only — **no Sentinel-1** in that bucket, so S1 still comes
from §2. DEA's STAC explorer was blocked from my web session, so navigate the
bucket by path or use STAC from a local session.

---

## 4. Copernicus Data Space (CDSE) — the source of record

Free registration, gives both S1 and S2 at every processing level, plus OData /
OpenSearch / STAC search and S3 access with generated credentials. Use it when
you need L1C, SLC, a specific baseline, or reprocessed archives. It was blocked
from my web session, so treat it as a local-session or notebook route.

---

## 5. Gotchas that cost me real time

1. **Tile-level cloud cover is nearly useless for a small AOI.** `eo:cloud_cover`
   describes ~12 000 km². I pulled a 14 km window from a tile reporting 0.3%
   cloud and got a frame half covered in white. Screen the *window*: read the
   same footprint out of `SCL.tif` (20 m) and count classes 3, 8, 9, 10 (shadow,
   cloud medium, cloud high, cirrus) plus class 0 for no-data.
2. **A tile that is "18% empty" can still cover your AOI completely.** Don't
   reject on tile-level nodata; check the window.
3. **MGRS tile names can be one latitude band off.** A point within a few km of a
   band edge converts to a square filed under the neighbouring band — 31.93°S
   converts to `54JWK` while the imagery lives under `54HWK`. Walk the ring of
   neighbouring tiles and keep the first whose raster *actually contains* the
   point. And assert the point is inside the raster before cutting, or you
   silently get a window clipped to the edge of the wrong tile.
4. **UTM hemisphere.** In `WGS 84 / UTM zone 55S` the `S` is the hemisphere, but
   it is also a valid *northern* MGRS band letter. Passing it to `utm.to_latlon`
   as a band letter puts a southern scene thousands of km north.
5. **For anything shadow-based, sun elevation is the whole ball game.** A 25°
   winter sun throws a 170 m shadow off an 80 m tower; a 65° summer sun throws
   37 m — under four pixels at 10 m. Ask for winter months and sort candidates by
   *lowest sun*, not lowest cloud.
6. **S1 is not a grey S2.** No shadows to measure, speckle everywhere, radar
   geometry until you warp, and DN that mean nothing until calibrated. Optical
   intuitions and optical algorithms do not carry over.

---

## 6. A working example to copy

`demo/fetch_scene.py` in this repository does the whole Sentinel-2 path end to
end: resolves the MGRS tile with the band-edge fallback, lists scenes, screens
candidates on the SCL window, picks the lowest sun, cuts the window, and writes
a GeoTIFF plus a `gdalinfo`-style sidecar carrying the per-band view angles.
`demo/README.md` has its command line and the three NSW scenes it was built on.

---

## 7. Prompt to paste into the other chat

> I want to pull Sentinel-2 (and possibly Sentinel-1) imagery for
> `<AREA / lat,lon>` over `<DATE RANGE>` and work with it in Python.
>
> Start by probing which endpoints this session can reach — in Claude Code on the
> web the public S3 buckets (`sentinel-cogs`, `sentinel-s1-l1c`,
> `dea-public-data`) usually work but the STAC APIs (earth-search, Planetary
> Computer, Copernicus) are blocked by the egress proxy. Use STAC search if it's
> reachable; otherwise navigate the bucket by MGRS tile path.
>
> For Sentinel-2 use the L2A COGs in `sentinel-cogs` and read only the window I
> need with rasterio (`Window`), never the whole 10980² band. Screen candidate
> scenes by reading the same window out of `SCL.tif` and counting cloud/shadow/
> cirrus/no-data — don't trust tile-level `eo:cloud_cover`. Watch for MGRS tiles
> that sit one latitude band away from where my point converts, and assert my
> point is inside the raster before cutting the window.
>
> For Sentinel-1, tell me first which of these I want: raw GRD from the bucket
> (radar geometry, needs `gdalwarp -tps` and calibration), ASF with an Earthdata
> login, or analysis-ready RTC from Planetary Computer.
