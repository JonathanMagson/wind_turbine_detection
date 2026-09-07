# Land-clearing detection from Sentinel-1 — NSW, Australia

Two runnable pipelines over New South Wales, both built on published,
peer-reviewed change-detection algorithms rather than new ones. Everything runs
server-side in Google Earth Engine — no bulk downloads, no SNAP preprocessing.

| Pipeline | Algorithm | Cover types | Use it when |
| --- | --- | --- | --- |
| [`omnibus_s1/`](omnibus_s1/) | Sequential omnibus Wishart test (Conradsen et al. 2016) | Forest, woodland | You want the highest-confidence woody clearing map, with dated changes, from radar alone |
| [`ccdc/`](ccdc/) | CCDC (Zhu & Woodcock 2014) on S1 + S2 stacks | Forest, woodland, **grassland** | You need one product across the whole cover gradient |

## Why two

The omnibus test is excellent on woody vegetation and degrades on grassland,
for a reason worth being explicit about. It assumes every image is an
independent sample from a *stationary* distribution. Woody canopy is dominated
by volume scattering and is close to stationary, which is why the method works
so well there. Grassland is not: it has a strong phenological cycle, and its
backscatter is driven by soil moisture — a rain front can move VV by 2–3 dB in
a week. A test that flags *any* departure fires constantly on things that are
not clearing.

CCDC fits each pixel's own harmonic season + trend model and flags departures
from *that*, with the threshold set by the pixel's own fitted RMSE. A noisy
grassland pixel gets a wider tolerance than a stable woodland pixel,
automatically. That is what makes one product across all three cover types
possible.

They are complementary, not redundant: where both fire on the same woody pixel,
you have two independent methods agreeing.

## The honest limit on grassland

**Sentinel-1 alone cannot establish grassland conversion, whatever the
algorithm.** Grassland has almost no volume scattering to lose. Ploughing
changes surface roughness, and the resulting backscatter change can go *either
way* depending on soil moisture, residue cover and row direction relative to
the look azimuth. The direction cue that makes woody clearing unambiguous is
simply absent.

So the CCDC pipeline requires Sentinel-2 evidence for grassland — an NDVI drop
*and* a bare-soil-index rise — and uses radar only as corroboration. Run it with
`--sensors s1` and grassland is reported as **zero**, which means *not
assessable*, not *no clearing*. The CLI warns when you do this.

## Layout

```
common/       AOI presets, woody masks and cover strata, area stats, validation
omnibus_s1/   omnibus.py (algorithm core, ported verbatim), s1.py, clearing.py, detect.py
ccdc/         collections.py (S1/S2 stacks), segmentation.py, rules.py, detect.py
notebooks/    nsw_omnibus_s1.ipynb, nsw_ccdc_multicover.ipynb
tests/        offline checks that need no Earth Engine credentials
```

## Setup

```bash
pip install -r land_clearing/requirements.txt
earthengine authenticate
```

You need an Earth Engine account and a Cloud project with the Earth Engine API
enabled. Run everything from the repository root.

## Run

Multi-cover (CCDC):

```bash
python -m land_clearing.ccdc.detect --aoi hay \
    --start 2023-01-01 --end 2024-01-01 --project your-gee-project
```

Woody only (omnibus):

```bash
python -m land_clearing.omnibus_s1.detect --aoi moree \
    --start 2023-01-01 --end 2024-01-01 --project your-gee-project
```

Both write a JSON summary and accept `--export drive` or
`--export asset --asset-folder projects/<p>/assets`.

### CCDC options worth knowing

| Flag | Default | Notes |
| --- | --- | --- |
| `--history-years` | `2.0` | History before `--start` used to fit the model. Below ~1.5 the harmonics are unreliable. |
| `--sensors` | `both` | `s1` cannot assess grassland; `s2` loses the direction constraint that makes woody detection specific. |
| `--min-observations` | `4` | Consecutive off-model observations to confirm a break. CCDC's own default of 6 assumes Landsat's 16-day revisit. |
| `--multilook-px` | `3` | Boxcar multi-look on S1. Speckle inflates the fitted RMSE, which raises the break threshold and costs sensitivity. |
| `--vh-drop-woody` | `-150` | VH magnitude threshold, dB × 100. So −1.5 dB. |
| `--ndvi-drop` / `--bsi-rise` | `-1500` / `800` | Index × 10000. So −0.15 NDVI and +0.08 BSI. |
| `--no-fire-screen` | off | By default, detections whose NBR fell past −0.30 are dropped as likely burns. |
| `--include-cropland` | off | Sown-crop cycles produce regular breaks that are not clearing. |
| `--min-mmu-ha` | `0.5` | Minimum mapping unit. Set `0` to see raw detections. |

Thresholds were chosen from the physical size of each effect, **not** tuned
against a reference. Calibrate them against SLATS for your AOI before quoting
numbers.

### AOI presets

`moree`, `walgett`, `collarenebri`, `narrabri` (Brigalow Belt South / north-west
cropping frontier), `pilliga` (woodland margin), `hay` (Riverina, grassland
dominated), `cobar` (western plains grassland / chenopod shrubland), `bombala`
(south-east forestry mix). Any other area: `--bbox W S E N`.

## Validation

```bash
python -m land_clearing.common.validation \
    --detected projects/<p>/assets/nsw_clearing_ccdc_hay_2023-01-01_2024-01-01 \
    --reference hansen --year 2023 --aoi hay
```

Hansen GFC is zero-setup but only meaningful for the **woody** strata — it is
tuned to closed-canopy forest, misses sparse woody cover, and says nothing
useful about grassland. Use it to sanity-check forest and woodland; ignore what
it implies about the rest.

For a defensible accuracy figure use **SLATS** (Statewide Landcover and Trees
Study) woody vegetation change from the NSW SEED portal at
<https://datasets.seed.nsw.gov.au/> — download the matching year, clip to the
AOI, upload as a GEE table asset, then pass `--reference asset
--reference-asset <table-asset>`. Note SLATS itself reports *woody* change, so
it validates the woody strata; grassland conversion has no equivalent
wall-to-wall NSW reference and needs field or high-resolution imagery checks.

## Known limitations

* **No radiometric terrain flattening.** `COPERNICUS/S1_GRD` is geometrically
  terrain corrected but not radiometrically flattened. Series over steep terrain
  are noisier and will over-detect. Flat inland AOIs behave best, which is also
  where most NSW clearing happens.
* **CCDC on radar is less trodden ground.** The algorithm and its GEE
  implementation are data-agnostic, but the published validation is
  overwhelmingly Landsat. Treat the S1 stack as the less-established half.
* **Fire.** The NBR screen is first-order. Separating fire from clearing
  properly needs a burnt-area product as an explicit exclusion layer.
* **Sparse woody baselines.** ESA WorldCover under-maps very sparse western NSW
  woody cover, so some woodland is stratified as grassland and then held to the
  stricter optical rule.
* **Timing resolution** is the acquisition cadence of the chosen S1 track
  (6–12 days) or the cloud-free S2 revisit, plus the `--min-observations`
  confirmation lag.

## Shortlist that was considered

| Repo | Method | Verdict for NSW |
| --- | --- | --- |
| [GEE CCDC](https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2020.576740/full), tooling in [`parevalo/gee-ccdc-tools`](https://github.com/parevalo/gee-ccdc-tools) | Harmonic season + trend, break on model residuals | **Chosen for multi-cover.** Built for land-cover change generally, self-calibrating per pixel, native to GEE and data-agnostic. |
| [`mortcanty/eesarseq`](https://github.com/mortcanty/eesarseq) | Sequential omnibus Wishart test | **Chosen for woody.** Peer-reviewed, unsupervised, dated and directed changes. Upstream has no LICENSE file, so this repo uses the Apache-2.0 tutorial source of the same algorithm. |
| [`bfast2`](https://github.com/bfast2) | Season + trend break monitoring | Same principle as CCDC and better established on SAR, but runs offline — you would pull per-pixel series out of GEE first. Better for a rigorous pixel study, worse for a whole-AOI demo. |
| [`jreiche/bayts`](https://github.com/jreiche/bayts) | Bayesian time-series fusion; basis of RADD alerts | Strongest deforestation-specific option, but R, and its forest/non-forest densities are parameterised for closed tropical forest. Woody only by construction. |
| [`ecovision-uzh/BraDD-S1TS`](https://github.com/ecovision-uzh/BraDD-S1TS) | Attention network on S1 time series | Trained on the Brazilian Amazon; applying it to NSW is a domain-shift research question. |
| [`bonlieuguillaume/vigisar`](https://github.com/bonlieuguillaume/vigisar) | Bi-temporal dissimilarity + tiled Otsu | Early-stage, no licence, bi-temporal only, expects GeoTIFFs you produce yourself. |
| [`CNES/S1Tiling`](https://github.com/CNES/S1Tiling) | S1 ARD preprocessing | Not a detector — the preprocessing layer the local-file options would need. |

## References

* Z. Zhu, C. E. Woodcock (2014). Continuous change detection and classification
  of land cover using all available Landsat data. *RSE* 144, 152–171.
* P. Arévalo et al. (2020). A Suite of Tools for Continuous Land Change
  Monitoring in Google Earth Engine. *Frontiers in Climate* 2, 576740.
* K. Conradsen, A. A. Nielsen, H. Skriver (2016). Determining the Points of
  Change in Time Series of Polarimetric SAR Data. *IEEE TGRS* 54(5), 3007–3024.
* Earth Engine Community tutorial, *Detecting Changes in Sentinel-1 Imagery*,
  parts 1–4 (Apache-2.0), <https://github.com/google/earthengine-community>.
