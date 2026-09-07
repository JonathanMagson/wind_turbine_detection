# Sentinel-1 land-clearing detection — NSW, Australia

A runnable demonstration of woody-vegetation clearing detection from Sentinel-1
SAR over New South Wales, built on a published, peer-reviewed change-detection
algorithm rather than a new one.

## Which algorithm, and why

**Chosen: sequential omnibus change detection for polarimetric SAR**
(Conradsen, Nielsen & Skriver 2016), in the Earth Engine implementation from the
[Detecting Changes in Sentinel-1 Imagery](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-3)
community tutorial series by Mort Canty, also packaged as
[`mortcanty/eesarseq`](https://github.com/mortcanty/eesarseq).

Why this one for a NSW demo:

* It runs entirely server-side in Earth Engine against `COPERNICUS/S1_GRD`, so
  there is no SLC download, no SNAP/pyroSAR preprocessing chain, and no orbit
  file wrangling. Whole-of-NSW coverage is a query, not a data-management
  project.
* It is a likelihood-ratio test on the complex Wishart distribution of the
  dual-pol covariance matrix — the change threshold is a significance level
  (`alpha`), not a tuned magic number, and it is calibrated by construction.
* It is *sequential*: it returns the timing and the direction of every
  significant change in the series, not just a before/after mask. Clearing dates
  fall out directly, which is what a regulatory or monitoring demo actually
  needs.
* No training data, so nothing to port from a tropical training domain.

### Shortlist that was considered

| Repo | Method | Verdict for NSW |
| --- | --- | --- |
| [`mortcanty/eesarseq`](https://github.com/mortcanty/eesarseq) | Sequential omnibus Wishart test, GEE | **Chosen.** Peer-reviewed, unsupervised, dated changes, zero data handling. Upstream is a notebook widget app with no LICENSE file, so this repo uses the Apache-2.0 tutorial source of the same algorithm. |
| [`jreiche/bayts`](https://github.com/jreiche/bayts) | Bayesian probabilistic time-series fusion (Reiche et al. 2015/2018); the basis of RADD alerts | Strongest *deforestation-specific* option and the obvious follow-up. R, and its forest/non-forest probability densities are parameterised for closed tropical forest, so NSW woodland needs re-parameterisation before the numbers mean anything. |
| [`ecovision-uzh/BraDD-S1TS`](https://github.com/ecovision-uzh/BraDD-S1TS) | Attention network on S1 time series | Trained on the Brazilian Amazon. Applying it to NSW sparse woody vegetation is a domain-shift research question, not a demo. |
| [`bonlieuguillaume/vigisar`](https://github.com/bonlieuguillaume/vigisar) | Bi-temporal dissimilarity + tiled Otsu thresholding | Early-stage (13 commits, no licence), bi-temporal only, and expects orthorectified GeoTIFFs you produce yourself. |
| [`chloeskt/SAR_deforestation_detection`](https://github.com/chloeskt/SAR_deforestation_detection) | SAR shadowing effect | Student coursework project; the shadowing cue needs tall closed canopy and steep look angles. |
| [`CNES/S1Tiling`](https://github.com/CNES/S1Tiling) | S1 ARD preprocessing | Not a detector — the preprocessing layer you would need for the local-file options above. |

## What this code adds

The omnibus test is a *generic* change detector. Three domain constraints turn
it into a clearing product (`nsw.py`):

1. **Direction.** Removing woody vegetation destroys volume scattering, so
   backscatter drops. Only negative-definite changes (`bmap == 2`) are kept.
2. **Baseline woody cover.** Clearing requires something to clear. Masking to a
   woody baseline removes most crop-cycle and soil-moisture false positives on
   cultivated paddocks.
3. **Minimum mapping unit.** An eight-connected component filter at 0.5 ha by
   default, comparable to the scale SLATS reports.

## Layout

```
omnibus.py    algorithm core, ported verbatim (Apache-2.0, attributed in-file)
s1.py         S1 GRD time-series construction (single relative orbit, linear power x ENL)
nsw.py        NSW AOI presets, woody masks, clearing post-processing, area stats
detect.py     end-to-end CLI, optional Drive/asset export
validate.py   confusion matrix and metrics against Hansen GFC or an uploaded reference
notebooks/nsw_land_clearing_demo.ipynb    Colab-ready walkthrough
```

## Setup

```bash
pip install -r requirements.txt
earthengine authenticate
```

You need an Earth Engine account and a Cloud project with the Earth Engine API
enabled.

## Run

```bash
cd land_clearing_s1
python detect.py --aoi moree --start 2023-01-01 --end 2024-01-01 \
    --project your-gee-project
```

Writes `clearing_summary.json` with the acquisition dates, the relative orbit
used, total detected clearing in hectares, and a per-interval breakdown. Add
`--export drive` (or `--export asset --asset-folder projects/<p>/assets`) to get
the raster out.

Useful knobs:

| Flag | Default | Notes |
| --- | --- | --- |
| `--orbit-pass` | `DESCENDING` | Try both; NSW coverage differs by track. |
| `--relative-orbit` | most frequent | Force a track for repeatability across runs. |
| `--alpha` | `0.01` | Per-test significance. Lower to `1e-3`/`1e-4` if maps look speckly. |
| `--median` | off | 5x5 median filter on the change maps; trades edge detail for noise. |
| `--woody-mask` | `worldcover` | `worldcover` \| `hansen` \| `palsar` \| `none`. |
| `--min-mmu-ha` | `0.5` | Set `0` to see raw detections. |
| `--stride` | `1` | Subsample the series to cut compute. |

### AOI presets

`moree`, `walgett`, `collarenebri`, `narrabri` (Brigalow Belt South / north-west
cropping frontier), `pilliga` (Pilliga–Nandewar woodland margin), `hay`
(Riverina mallee/box woodland edge), `bombala` (south-east forestry and clearing
mix). Any other area: `--bbox W S E N`.

## Validation

```bash
python validate.py --detected projects/<p>/assets/nsw_clearing_moree_2023-01-01_2024-01-01 \
    --reference hansen --year 2023 --aoi moree
```

Hansen GFC is zero-setup but a weak yardstick here: it is tuned to closed-canopy
forest and systematically misses the sparse woody vegetation that most NSW
clearing removes, so it will make recall look worse than it is.

For a defensible accuracy figure use **SLATS** (Statewide Landcover and Trees
Study) woody vegetation change, published on the NSW SEED portal at
<https://datasets.seed.nsw.gov.au/> — download the matching year, clip to the
AOI, upload as a GEE table asset, then:

```bash
python validate.py --detected <image-asset> \
    --reference asset --reference-asset <table-asset> --aoi moree
```

## Known limitations

* **No radiometric terrain flattening.** `COPERNICUS/S1_GRD` is geometrically
  terrain corrected but not radiometrically flattened. The test is per-pixel and
  purely temporal so this is tolerable, but series over steep terrain (Great
  Dividing Range, Blue Mountains) are noisier and will over-detect. Flat inland
  AOIs behave best, which is also where most NSW clearing happens.
* **Backscatter drops are not uniquely clearing.** Harvest, fire, flooding
  drawdown and drought dieback all reduce backscatter. The woody mask and MMU
  suppress most of it; separating fire from clearing needs a burnt-area layer as
  an additional exclusion, which is not implemented here.
* **Sparse woody baselines.** WorldCover under-maps very sparse western NSW
  woody cover. Compare `--woody-mask palsar` (radar-derived, closest to what S1
  senses) against the default before quoting numbers.
* **Timing resolution** is the acquisition interval of the chosen track: 6–12
  days depending on the period and whether S1B/S1C data are present.

## References

* K. Conradsen, A. A. Nielsen, H. Skriver (2016). Determining the Points of
  Change in Time Series of Polarimetric SAR Data. *IEEE TGRS* 54(5), 3007–3024.
* Earth Engine Community tutorial, *Detecting Changes in Sentinel-1 Imagery*,
  parts 1–4 (Apache-2.0), <https://github.com/google/earthengine-community>.
* J. Reiche et al. (2018). Characterizing Tropical Forest Cover Loss Using Dense
  Sentinel-1 Data and Active Fire Alerts. *Remote Sensing* 10(5), 777.
