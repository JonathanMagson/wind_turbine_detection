#!/usr/bin/env python3
"""Render figures from a local_s1 run.

    python -m land_clearing.local_s1.figures out_moree_2023
"""

import json
import os
import sys

import numpy as np


def _db(sigma0):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.where(sigma0 > 0, sigma0, np.nan))


def render(outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    z = np.load(os.path.join(outdir, 'arrays.npz'))
    summary = json.load(open(os.path.join(outdir, 'summary.json')))
    dates = summary['dates']
    bbox = summary['bbox']
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

    first_neg = z['first_neg']
    woody = z['woody']
    detected = first_neg > 0

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    fig.suptitle(
        'Sentinel-1 land-clearing detection, NSW  |  %s to %s  |  '
        'relative orbit %d, %d acquisitions'
        % (dates[0], dates[-1], summary['relative_orbit'], summary['n_scenes']),
        fontsize=13)

    ax = axes[0, 0]
    ax.imshow(_db(z['vh_first']), cmap='gray', vmin=-26, vmax=-14, extent=extent)
    ax.set_title('VH sigma0 (dB), first acquisition %s' % dates[0], fontsize=10)

    ax = axes[0, 1]
    ax.imshow(_db(z['vh_last']), cmap='gray', vmin=-26, vmax=-14, extent=extent)
    ax.set_title('VH sigma0 (dB), last acquisition %s' % dates[-1], fontsize=10)

    ax = axes[1, 0]
    ax.imshow(woody, cmap=ListedColormap(['#f2efe6', '#2e7d32']), extent=extent)
    ax.set_title('WorldCover woody baseline (tree + shrub), %.0f%% of AOI'
                 % (100.0 * woody.mean()), fontsize=10)

    ax = axes[1, 1]
    ax.imshow(_db(z['vh_first']), cmap='gray', vmin=-26, vmax=-14, extent=extent)
    n_int = max(1, len(dates) - 1)
    if detected.any():
        masked = np.ma.masked_where(~detected, first_neg)
        cmap = plt.get_cmap('autumn_r', n_int)
        im = ax.imshow(masked, cmap=cmap, vmin=1, vmax=n_int, extent=extent)
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label('interval of first backscatter drop', fontsize=8)
    ax.set_title('Detected clearing: %.1f ha (%.2f%% of woody area)'
                 % (summary['detected_clearing_ha'],
                    100.0 * summary['detected_clearing_ha']
                    / max(summary['woody_area_ha'], 1e-9)), fontsize=10)

    for ax in axes.ravel():
        ax.set_xlabel('longitude', fontsize=8)
        ax.set_ylabel('latitude', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    p1 = os.path.join(outdir, 'overview.png')
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # Timeline of detected area per interval.
    per = summary.get('per_interval', {})
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = list(range(1, len(dates)))
    ys = [per.get(str(i), per.get(i, {})).get('hectares', 0.0) for i in xs]
    ax.bar(xs, ys, color='#a50f15')
    ax.set_xticks(xs)
    ax.set_xticklabels(['%s' % dates[i][5:] for i in xs], rotation=90, fontsize=7)
    ax.set_xlabel('interval end date (%s)' % dates[0][:4], fontsize=9)
    ax.set_ylabel('detected clearing (ha)', fontsize=9)
    ax.set_title('When the detections occurred', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(outdir, 'timeline.png')
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return [p1, p2]


if __name__ == '__main__':
    for p in render(sys.argv[1] if len(sys.argv) > 1 else 'out'):
        print(p)
