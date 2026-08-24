# Turbine Shadow Census — an in-browser demo

A one-command visual for the a-contrario wind turbine detector in this
repository. The Python side runs the published `main_NFA` detector on a
Sentinel-2 band; the browser side lets you push the significance threshold
around, zoom into any pixel and watch the two tests that produced its score.

```
python3 -m venv .venv
.venv/bin/pip install -r demo/requirements.txt
.venv/bin/python demo/turbine_demo.py --serve
```

That detects turbines in the bundled 23 × 15 km scene (`band2_masked.tif`,
southern New South Wales, April 2024), writes `demo/out/index.html` and opens
it. Without `--serve` you get the same page as a single self-contained file you
can mail to someone.

## What the page does

| Panel | What it shows |
| --- | --- |
| Map | The reflectance band, the −log₁₀(NFA) evidence field, the above-cut mask and one marker per candidate. Wheel to zoom, drag to pan, click anything. |
| Detection cut | The significance threshold, re-thresholded and re-clustered **in the browser** — the full 16-bit NFA map travels with the page. |
| Acquisition geometry | Sun and satellite angles from the metadata, drawn in plan and in profile, with the modelled shadow length and hub offset. |
| Evidence distribution | Where the scene's pixels sit on the −log₁₀(NFA) axis, and where your cut falls. |
| Why this pixel | The selected pixel's 13 shadow samples and 7 hub samples, each marked pass or fail, its K out of 20, the binomial tail and the resulting NFA — recomputed from the pixel values in the browser, and equal to the number on the map. |

With `--serve`, the shadow and hub contrast thresholds (`diff_s`, `diff_h`) are
live too: moving either slider re-runs `main_NFA` on the server (~9 s for the
full scene) and streams a new evidence map back into the canvas.

Every run also writes `detections.csv` and `detections.geojson` next to the
page, so candidates open straight in QGIS.

## Options

```
--image PATH          single-band Sentinel-2 GeoTIFF (default band2_masked.tif)
--meta PATH           gdalinfo-style sidecar for that image
--band B02            which band's viewing angles to use
--heights 60,80,100   hub heights to test; the map keeps the best NFA
--t-nfa 1             -log10(NFA) cut used for the CSV/GeoJSON and the first view
--t-shadow 25         shadow contrast threshold, in digital numbers
--t-hub 50            hub contrast threshold, in digital numbers
--sun-time auto       which metadata timestamp fixes the sun position
--out DIR             where to write index.html and the sidecars
--serve --port 8000   serve the page with the live detector attached
--fragment            also write a head-less HTML fragment for embedding
```

## Three things worth knowing about the data path

**The UTM hemisphere comes from the projection, not the zone letter.** In
`WGS 84 / UTM zone 55S` the trailing `S` means southern hemisphere, but it is
also a valid MGRS latitude-band letter in the *northern* hemisphere — feeding it
to `utm.to_latlon` as a band letter puts the bundled scene about 8000 km north
of where it is, and drags the computed sun position with it. `demo/pipeline.py`
parses the hemisphere from the `PROJCRS` name and passes `northern=` explicitly.
It also ignores the sidecar's `epsg=` tag, which is stale in `band2_metadata.txt`
(it names zone 36 for a zone 55 scene).

**`granule_date` can disagree with `date`.** `band2_metadata.txt` carries
`granule_date=2024-04-23 07:00:35`, which puts the sun 4.5° above the horizon —
nearly sunset, with the modelled shadow pointing the wrong way. Its `date` tag
gives 00:02 UTC and a 33° sun, which matches a mid-morning Sentinel-2 pass at
149° E. In `--sun-time auto` the demo keeps `granule_date` unless it puts the sun
below 10° and the other tag puts it higher; `--sun-time granule_date` forces the
original behaviour, which is a good way to watch the detector fail on wrong
geometry.

**The hub test compares against the tested pixel's neighbours.** In `main_NFA`
the six hub comparison points are `plagex + compx1` — a ring around the pixel
under test, not around each hub sample. The browser's replay mirrors that
exactly; that is why its recomputed NFA equals the map's to the last digit.

## Layout

```
demo/pipeline.py      scene + metadata parsing, geometry, detection, raster packing
demo/turbine_demo.py  CLI: builds the page, optionally serves it with a live detector
demo/template.html    the page itself (styles, markup, viewer); __PAYLOAD__ is injected
demo/requirements.txt floating dependencies for a current Python
```

The detector itself is untouched: `pipeline.py` calls
`utils_eol_compute.main_NFA`, `main_shadow_S2` and `main_hub_S2` from the
repository root, so the visual can only ever show what the published algorithm
does.

Detector © Nicolas Mandroux, v0.1 (2021), AGPL-3.0 — see the root `README.txt`.
Imagery: Copernicus Sentinel-2, ESA.
