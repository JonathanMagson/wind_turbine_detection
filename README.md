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

See [`demo/README.md`](demo/README.md) for the options and for three metadata
gotchas the demo handles (UTM hemisphere parsing, disagreeing acquisition
timestamps, and the hub test's comparison points).
