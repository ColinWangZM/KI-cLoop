import argparse
import json
from dataclasses import replace
from pathlib import Path
import runtime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kinetic_core import predict
from run import ROOT, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load()
    reference = json.loads((ROOT / 'reference_results/seed_comparisons.json').read_text())['20261009']
    q = np.array(reference['ODE_refined']['q'])
    dense = replace(data, times=np.broadcast_to(np.linspace(0, data.times.max(), 240), (len(data.times), 240)).copy())
    concentrations = predict(dense, q, .5)[:, :, 0]
    fig, axes = plt.subplots(6, 5, figsize=(13, 13), sharex=True, sharey=True)
    ordered = sorted(range(len(data.times)), key=lambda i: (data.conditions[i, 0], data.initial[i, 0]))
    for ax, index in zip(axes.ravel(), ordered):
        observed = 100 * (1 - data.observations[index, :, 0] / data.initial[index, 0])
        modeled = 100 * (1 - concentrations[index] / data.initial[index, 0])
        ax.plot(dense.times[index], modeled, color='#2367a3', lw=1.6)
        ax.scatter(data.times[index, 1:], observed[1:], s=17, color='#d67824', zorder=3)
        ax.scatter([0], [0], marker='s', s=14, facecolors='none', edgecolors='black')
        ax.set_title(f'{data.conditions[index, 0] - 273.15:g} °C; {data.initial[index, 0]:.1f}/{data.initial[index, 1]:.1f} M', fontsize=9)
        ax.set_ylim(-3, 105)
        ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='#2367a3', label='Final ODE'),
                        Line2D([0], [0], marker='o', linestyle='', color='#d67824', label='Measured HXD conversion'),
                        Line2D([0], [0], marker='s', linestyle='', markerfacecolor='none', color='black', label='Prescribed inlet')],
               loc='upper center', bbox_to_anchor=(.5, .977), ncol=3, frameon=False)
    fig.supxlabel('Residence time (s)')
    fig.supylabel('HXD conversion (%)')
    fig.suptitle('Paal–Knorr: observations and final ODE trajectories', y=.998)
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.savefig(args.output / 'all_conditions.png', dpi=180)
    plt.close(fig)
    frame = pd.read_csv(ROOT / 'reference_results/predictions.csv')
    frame = frame[frame.independent_fit_observation]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharex=True, sharey=True)
    columns = [('final_ODE_conversion_pct', 'Final hybrid fit'),
               ('leave_temperature_out_ODE_conversion_pct', 'Leave-temperature-out ODE'),
               ('leave_ratio_out_ODE_conversion_pct', 'Leave-ratio-out ODE')]
    actual = frame.observed_conversion_pct.to_numpy()
    for ax, (column, title) in zip(axes, columns):
        prediction = frame[column].to_numpy()
        errors = prediction - actual
        r2 = 1 - np.sum(errors ** 2) / np.sum((actual - actual.mean()) ** 2)
        rmse = np.sqrt(np.mean(errors ** 2))
        points = ax.scatter(actual, prediction, c=frame.temperature_C, cmap='viridis', s=17, alpha=.8)
        ax.plot([0, 100], [0, 100], color='gray', lw=1)
        ax.set_title(title, fontsize=10)
        ax.text(.04, .96, f'R² = {r2:.4f}\nRMSE = {rmse:.3f} pp', transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.set_xlabel('Observed HXD conversion (%)')
    axes[0].set_ylabel('Predicted HXD conversion (%)')
    fig.subplots_adjust(left=.07, right=.9, bottom=.18, top=.9, wspace=.14)
    colorbar_axes = fig.add_axes([.93, .22, .012, .63])
    fig.colorbar(points, cax=colorbar_axes, label='Temperature (°C)')
    fig.savefig(args.output / 'fit_and_grouped_validation.png', dpi=180)
    plt.close(fig)
    print('Wrote all_conditions.png and fit_and_grouped_validation.png')


if __name__ == '__main__':
    main()
