# wind_turbine_detection

Wind turbine a-contrario detection on Sentinel-2 images — see `README.txt` for
the original algorithm description, dependencies and citation.

## Browser demo

`demo/` adds an interactive visual on top of the published detector: it runs
`main_NFA` on a scene, then hands you a page where you can move the
significance cut, zoom to a single pixel and see the shadow and hub tests that
scored it.

```
python3 -m venv .venv
.venv/bin/pip install -r demo/requirements.txt
.venv/bin/python demo/turbine_demo.py --serve
```

`demo/fetch_scene.py` cuts a fresh window out of the public Sentinel-2 archive
if you want to point the detector somewhere else — no credentials needed:

```
.venv/bin/pip install -r demo/requirements-fetch.txt
.venv/bin/python demo/fetch_scene.py --lat -34.5753 --lon 148.8701 --km 14 \
    --year 2024 --months 5,6,7,8 --name ryepark
```

See [`demo/README.md`](demo/README.md) for the options and for the metadata
gotchas both scripts handle (UTM hemisphere parsing, disagreeing acquisition
timestamps, the hub test's comparison points, window-level cloud screening and
MGRS tiles that sit one latitude band away from where a point converts).
