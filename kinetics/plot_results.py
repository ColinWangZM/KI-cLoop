import runtime
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import replace
from kinetic_core import HERE, CONFIGS, load_data, predict

def main():
    parser = argparse.ArgumentParser(description='Plot measured trajectories, model fits and activation-energy uncertainty.')
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (key, cfg) in zip(axes, CONFIGS.items()):
        data = load_data(key)
        final = json.loads((args.results / key / 'final_summary.json').read_text())
        pred = predict(data, np.array(final['q']), final['step'])
        index = data.species.index('TBPB' if key == 'tbpb_benzaldehyde' else 'TBHP')
        observed = data.observations[:, 1:, index].ravel()
        estimate = pred[:, 1:, index].ravel()
        ax.scatter(observed, estimate, s=14, alpha=0.7)
        extent = max(observed.max(), estimate.max()) * 1.03
        ax.plot([0, extent], [0, extent], color='black', lw=0.8)
        ax.set(xlabel='Observed (mol/L)', ylabel='Predicted (mol/L)', title=f"Route {cfg['route_number']}")
        n = len(data.times)
        fig, axs = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(12, 2.5 * np.ceil(n / 4)), squeeze=False)
        grid = np.linspace(0, 1, 201)
        dense = replace(data, times=data.times[:, -1:] * grid[None, :])
        curves = predict(dense, np.array(final['q']), final['step'])
        for e, sub in enumerate(axs.flat):
            if e >= n:
                sub.axis('off')
                continue
            for s, name in enumerate(data.species):
                line, = sub.plot(dense.times[e], curves[e, :, s], label=name)
                sub.scatter(data.times[e, 1:], data.observations[e, 1:, s], s=12, color=line.get_color())
            sub.set(title=data.sheets[e], xlabel='Residence time (s)', ylabel='Concentration (mol/L)')
        axs.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output / f'{key}_trajectories.pdf')
        plt.close(fig)
    figure.tight_layout()
    figure.savefig(args.output / 'product_parity.pdf')
    plt.close(figure)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, key in enumerate(CONFIGS):
        u = json.loads((args.results / key / 'uncertainty.json').read_text())['intervals']['Ea1_kJ_mol']
        ax.errorbar(i + 1, u['estimate'], yerr=[[u['estimate'] - u['low']], [u['high'] - u['estimate']]], fmt='o', capsize=4)
    ax.set(xticks=[1, 2, 3, 4], xlabel='Route', ylabel='Apparent activation energy (kJ/mol)')
    fig.tight_layout()
    fig.savefig(args.output / 'activation_energy_intervals.pdf')
    plt.close(fig)
if __name__ == '__main__':
    main()
