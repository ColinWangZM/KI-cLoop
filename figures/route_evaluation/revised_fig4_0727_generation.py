
from __future__ import annotations
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from IPython.display import Image as IPImage, display

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KINETICS_ROOT = Path(
    os.environ.get('KGRAG_KINETICS_PIPELINE', PROJECT_ROOT / 'data/private/kinetics_pipeline')
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get('KGRAG_KINETICS_OUTPUTS', KINETICS_ROOT / 'outputs')
).expanduser()
FIG_ROOT = Path(__file__).resolve().parent
PANEL_ROOT = FIG_ROOT / 'panels'
COMPOSITE_ROOT = FIG_ROOT / 'composite'
PANEL_ROOT.mkdir(parents=True, exist_ok=True)
COMPOSITE_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(KINETICS_ROOT))
from route_evaluation import ROUTES, write_outputs

write_outputs()

design = pd.read_csv(OUTPUT_ROOT / 'route_experimental_design_summary.csv')
measured = pd.read_csv(OUTPUT_ROOT / 'route_measured_candidate_pool.csv')
pred_pool = pd.read_csv(OUTPUT_ROOT / 'route_product_prediction_pool.csv')
condition_cmp = pd.read_csv(OUTPUT_ROOT / 'route_condition_comparison.csv')
kin_params = pd.read_csv(OUTPUT_ROOT / 'route_kinetic_parameter_summary.csv')
route_eval = pd.read_csv(OUTPUT_ROOT / 'route_evaluation_summary.csv')

route_keys = ['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3', 'tbpb_benzaldehyde']
route_order = ['Route 1', 'Route 2', 'Route 3', 'Route 4']
route_colors = {'Route 1': '#4C78A8', 'Route 2': '#59A14F', 'Route 3': '#F28E2B', 'Route 4': '#B07AA1'}
main_color = '#2F6F9F'
side_color = '#D8893A'

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    # Figures are placed as half-width panels in the manuscript.  Keep every
    # annotation readable at that final printed size.
    'font.size': 11.0,
    'axes.titlesize': 13.0,
    'axes.labelsize': 11.0,
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'axes.linewidth': 0.65,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})

def finish_axes(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=2.3, width=0.6, pad=1.4)
    if grid_axis:
        ax.grid(axis=grid_axis, color='#D9DEE2', lw=0.45, alpha=0.48)

def save_panel(fig, name):
    path = PANEL_ROOT / name
    # Preserve the declared canvas ratio.  The companion PPT layout uses the
    # same ratio for each image, so no text, symbols, or data marks are ever
    # stretched during placement.  Avoid bbox_inches='tight': its content-
    # dependent crop changes the exported PNG aspect ratio panel by panel.
    fig.savefig(path, facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    display(IPImage(filename=str(path)))
    return path

def panel_label(fig, letter, title, title_size=14.0):
    fig.text(0.01, 0.99, letter, ha='left', va='top', fontsize=17, fontweight='bold')
    fig.text(0.065, 0.985, title, ha='left', va='top', fontsize=title_size, fontweight='bold')

def route_training(route_key):
    return pd.read_csv(OUTPUT_ROOT / f'{route_key}_ode_training_data.csv')

def ratio_for_route_df(route_key, df, suffix='_0'):
    if route_key == 'tbhp_bzcl':
        return df[f'H2O2{suffix}'] / df[f'TBCL{suffix}'].replace(0, np.nan)
    if route_key in ['tbhp_wpo4', 'tbhp_cf3']:
        return df[f'H2O2{suffix}'] / df[f'TBA{suffix}'].replace(0, np.nan)
    return df[f'TBHP{suffix}'] / df[f'Benzaldehyde{suffix}'].replace(0, np.nan)


# %%

# Fig. 4a: four pathway profiles. Each trace is an original measured
# condition family, not an interpolated or model-generated response curve.
# A 2 × 2 internal layout keeps this panel balanced with Fig. 4b in the
# requested two-column manuscript arrangement.
fig = plt.figure(figsize=(7.2, 4.33))
axes = [
    fig.add_axes([0.13, 0.62, 0.36, 0.23]),
    fig.add_axes([0.58, 0.62, 0.36, 0.23]),
    fig.add_axes([0.13, 0.19, 0.36, 0.23]),
    fig.add_axes([0.58, 0.19, 0.36, 0.23]),
]
panel_label(fig, 'a', 'Measured condition-response profiles', title_size=14.0)
fig.text(0.99, 0.965, '● Cat./acid     ▲ Feed ratio     ■ Temperature',
         ha='right', va='top', fontsize=10.5, color='#333333')
category_styles = {
    'Cat./acid': {'marker': 'o', 'linestyle': '-', 'row_label': 'Cat.'},
    # The three condition families are encoded by marker shape only.  All
    # response trajectories remain solid lines for a clean, unambiguous style.
    'Ratio': {'marker': '^', 'linestyle': '-', 'row_label': 'Ratio'},
    'Temp.': {'marker': 's', 'linestyle': '-', 'row_label': 'Temp.'},
}
profile_specs = {
    'tbhp_bzcl': [
        ('Cat./acid', [], []),
        ('Ratio', list(range(0, 6)), ['1.00', '2.00', '3.00', '4.00', '5.00', '6.00']),
        ('Temp.', list(range(6, 13)), ['30', '35', '40', '45', '50', '55', '60']),
    ],
    'tbhp_wpo4': [
        ('Cat./acid', list(range(0, 5)), ['0.05', '0.09', '0.12', '0.14', '0.17']),
        ('Ratio', list(range(5, 10)), ['1.20', '1.40', '1.60', '1.80', '2.00']),
        ('Temp.', list(range(10, 15)), ['60', '65', '70', '75', '80']),
    ],
    'tbhp_cf3': [
        ('Cat./acid', list(range(0, 6)), ['1.40', '1.80', '2.10', '2.50', '2.90', '3.20']),
        ('Ratio', list(range(6, 11)), ['1.20', '1.40', '1.60', '1.80', '2.00']),
        ('Temp.', list(range(11, 17)), ['60', '65', '70', '75', '80', '85']),
    ],
    'tbpb_benzaldehyde': [
        ('Cat./acid', list(range(0, 5)), ['0.14', '0.21', '0.28', '0.36', '0.43']),
        ('Ratio', list(range(5, 10)), ['1.10', '1.40', '1.70', '2.00', '2.30']),
        ('Temp.', list(range(10, 15)), ['45', '50', '55', '60', '65']),
    ],
}

def select_five(ids, labels):
    """Use the previously approved five-column condition ledger layout."""
    if len(ids) == 0:
        return [], []
    if len(ids) <= 5:
        return ids, labels
    idx = np.linspace(0, len(ids) - 1, 5).round().astype(int)
    idx = np.unique(idx)
    while len(idx) < 5:
        for candidate in range(len(ids)):
            if candidate not in idx:
                idx = np.append(idx, candidate)
                if len(idx) == 5:
                    break
    idx = np.sort(idx[:5])
    return [ids[i] for i in idx], [labels[i] for i in idx]


for panel_index, (ax, route_key, route_label) in enumerate(zip(axes, route_keys, route_order)):
    route_measured = measured[measured['route'] == route_key]
    exp_max = route_measured.groupby('experiment_id')['product'].max()
    route_max = max(float(exp_max.max()), 1e-12)
    ax.set_title(route_label.replace('Route', 'Path'), color=route_colors[route_label],
                 fontsize=12.0, pad=2, fontweight='bold')
    # Keep the previously approved five-column layout beneath each profile.
    # The only Path 1-specific addition is its explicit catalyst slash row.
    max_levels = 5
    for row_idx, (cat_name, exp_ids_all, labels_all) in enumerate(profile_specs[route_key]):
        style = category_styles[cat_name]
        y_text = -0.22 - row_idx * 0.175
        if not exp_ids_all:
            # Path 1 has no catalyst variable. Retain slash entries explicitly
            # so its condition ledger remains structurally aligned.
            if panel_index in (0, 2):
                ax.text(-0.34, y_text, style['row_label'], transform=ax.get_xaxis_transform(),
                        ha='right', va='center', fontsize=9.5, color='#444444', clip_on=False)
            for x_pos in range(max_levels):
                ax.text(x_pos, y_text, '/', transform=ax.get_xaxis_transform(),
                        ha='center', va='center', fontsize=9.7, color='#555555', clip_on=False)
            continue
        exp_ids, labels = select_five(exp_ids_all, labels_all)
        x = np.arange(len(exp_ids))
        vals = np.array([exp_max.get(i, np.nan) for i in exp_ids], dtype=float) / route_max
        ax.plot(x, vals, marker=style['marker'], ms=5.3, lw=1.80, color=route_colors[route_label],
                markeredgecolor='white', markeredgewidth=0.55, zorder=3,
                linestyle=style['linestyle'], alpha=0.98)
        # Directly link each line style to its true experimental condition values.
        if panel_index in (0, 2):
            ax.text(-0.34, y_text, style['row_label'], transform=ax.get_xaxis_transform(),
                    ha='right', va='center', fontsize=9.5, color='#444444', clip_on=False)
        for x_pos, label in zip(x, labels):
            ax.text(x_pos, y_text, label, transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=9.2, color='#333333', clip_on=False)
    ax.set_xlim(-0.40, max_levels - 0.60)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(max_levels), [''] * max_levels)
    finish_axes(ax)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis='y', labelsize=9.3)
    if panel_index % 2:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Normalized product\n(max = 1)', fontsize=11.5)
panel_a = save_panel(fig, 'fig4a_normalized_condition_profiles.png')

# %%

# Fig. 4b: representative measured time-course profiles at the best experimental condition
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.33), sharey=False)
axes = axes.ravel()
panel_label(fig, 'b', 'Best-condition time-course')

profile_species = {
    'tbhp_bzcl': [('TBHP', 'target')],
    'tbhp_wpo4': [('TBHP', 'target'), ('DTBP', 'side')],
    'tbhp_cf3': [('TBHP', 'target'), ('DTBP', 'side')],
    'tbpb_benzaldehyde': [('TBPB', 'target')],
}

def condition_label(route_key, row):
    if route_key == 'tbhp_bzcl':
        ratio = row['H2O2_0'] / row['TBCL_0']
        return f"ratio={ratio:.2f}, T={row['T_0']:.0f}°C, cat=/"
    if route_key in ['tbhp_wpo4', 'tbhp_cf3']:
        ratio = row['H2O2_0'] / row['TBA_0']
        return f"ratio={ratio:.2f}, T={row['T_0']:.0f}°C, cat={row['Catalyst_0']:.2f}"
    ratio = row['TBHP_0'] / row['Benzaldehyde_0']
    return f"ratio={ratio:.2f}, T={row['T_0']:.0f}°C, cat={row['Catalyst_0']:.2f}"

best_ids = {}
for route_key, route_label in zip(route_keys, route_order):
    route_rows = measured[(measured['route'] == route_key) & (measured['time'] > 0)].copy()
    by_family = route_rows.groupby('experiment_id')['product'].max().sort_values(ascending=False)
    best_ids[route_key] = int(by_family.index[0])

for panel_index, (ax, route_key, route_label) in enumerate(zip(axes, route_keys, route_order)):
    df = route_training(route_key)
    sub = df[df['experiment_id'] == best_ids[route_key]].sort_values('time')
    route_color = route_colors[route_label]
    for species, role in profile_species[route_key]:
        if species not in sub.columns:
            continue
        if role == 'target':
            line_color, ls, lw, marker, alpha = route_color, '-', 1.90, 'o', 1.0
        else:
            line_color, ls, lw, marker, alpha = '#8A8A8A', '--', 1.15, 's', 0.85
        ax.plot(sub['time'], sub[species], color=line_color, lw=lw, ls=ls, marker=marker, ms=4.8, alpha=alpha, label=species)
    ax.set_title(route_label.replace('Route', 'Path'), color=route_color, pad=3, fontsize=13.0, fontweight='bold')
    # Bottom-row labels only: repeating x-axis titles in all four small axes
    # competes with the Path 3/4 headings in the compact 2 × 2 layout.
    if panel_index >= 2:
        ax.set_xlabel('Time (s)', fontsize=11.0)
    else:
        ax.set_xlabel('')
    ax.set_xlim(left=0)
    ymax = max(float(sub[[s for s, _ in profile_species[route_key] if s in sub.columns]].max().max()), 1e-12)
    if route_key == 'tbpb_benzaldehyde':
        ax.set_ylim(0, ymax * 1.36)
    else:
        ax.set_ylim(0, ymax * 1.18)
    finish_axes(ax)
    # Put the condition above the plotting area so it cannot cover a data line.
    ax.text(0.03, 0.95, condition_label(route_key, sub.iloc[0]), transform=ax.transAxes,
            fontsize=9.8, color='#333333', ha='left', va='top')
    ax.tick_params(labelsize=10.0)
    if panel_index < 2:
        ax.tick_params(axis='x', labelbottom=False)
axes[0].set_ylabel('Measured concentration\n(mol L$^{-1}$)', fontsize=12.0)
axes[2].set_ylabel('Measured concentration\n(mol L$^{-1}$)', fontsize=12.0)
legend_line = mpl.lines.Line2D([0.60, 0.67], [0.035, 0.035], transform=fig.transFigure, color='#8A8A8A', lw=1.4, ls='--')
fig.add_artist(legend_line)
fig.text(0.685, 0.035, 'Side reaction', ha='left', va='center', fontsize=10.0, color='#222222')
fig.subplots_adjust(left=0.135, right=0.985, bottom=0.13, top=0.86, hspace=0.36, wspace=0.30)
# The two left axes have different tick-label widths.  Pin both ylabel
# anchors to the same axes coordinate so their vertical titles align exactly.
for ax in (axes[0], axes[2]):
    ax.yaxis.set_label_coords(-0.10, 0.50)
panel_b = save_panel(fig, 'fig4b_best_condition_timecourse.png')

# %%

# Fig. 4c: route-wise PINN metrics calculated directly from the original
# experimental/prediction pairs.  No values are transcribed from a figure.
metric_rows = []
for route in route_order:
    sub = pred_pool[pred_pool['route_label'] == route]
    observed = sub['experimental'].to_numpy(float)
    predicted = sub['predicted'].to_numpy(float)
    residual = predicted - observed
    ss_total = float(np.sum((observed - observed.mean()) ** 2))
    metric_rows.append({
        'route_label': route,
        'R2': 1 - float(np.sum(residual ** 2)) / ss_total,
        'MAE': float(np.mean(np.abs(residual))),
        'MSE': float(np.mean(residual ** 2)),
    })
metric_df = pd.DataFrame(metric_rows).set_index('route_label').loc[route_order]

fig = plt.figure(figsize=(7.2, 4.33))
panel_label(fig, 'c', 'PINN prediction accuracy')
metric_specs = [
    ('R2', r'$R^2$', r'$\mathrm{unitless}$', (0.90, 1.00), [0.90, 0.95, 1.00], '.2f'),
    ('MAE', 'MAE', r'$\mathrm{mol\ L^{-1}}$', (0.00, 0.10), [0.00, 0.05, 0.10], '.3f'),
    ('MSE', 'MSE', r'$\mathrm{mol^{2}\ L^{-2}}$', (0.00, 0.020), [0.00, 0.01, 0.02], '.4f'),
]
for col_index, (key, title, unit, limits, ticks, fmt) in enumerate(metric_specs):
    # A three-column metric layout is intentionally matched to the wide
    # c-panel canvas; it avoids the crowded stacked-axis labels of the narrow
    # layout while retaining all values calculated from original predictions.
    left = [0.145, 0.435, 0.725][col_index]
    ax = fig.add_axes([left, 0.18, 0.23, 0.67])
    for row in [0, 2]:
        ax.axhspan(row - 0.43, row + 0.43, color='#F2F5F6', zorder=0)
    for row, route in enumerate(route_order):
        value = metric_df.loc[route, key]
        color = route_colors[route]
        ax.hlines(row, limits[0], value, color=color, lw=1.1, alpha=0.60)
        ax.scatter(value, row, s=64, color=color, edgecolor='white', linewidth=0.7, zorder=3)
        ax.annotate(format(value, fmt), (value, row), xytext=(-3, 5),
                    textcoords='offset points', ha='right', va='bottom',
                    fontsize=10.5, color='#222222')
    ax.set(xlim=limits, ylim=(3.45, -0.55), xticks=ticks)
    ax.set_yticks(range(4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#D9DEE2', lw=0.55)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.tick_params(axis='x', length=2.2, width=0.6, pad=2, labelsize=9.5)
    ax.set_title(title, loc='left', fontsize=13.0, pad=3, fontweight='bold')
    # Plain unicode superscripts and one shared baseline keep all three units
    # optically aligned despite their differing exponents.
    fig.text(left + 0.115, 0.090, unit, fontsize=10.2,
             ha='center', va='baseline', color='#555555')
    if col_index == 0:
        ax.set_yticklabels(route_order)
        for label, route in zip(ax.get_yticklabels(), route_order):
            label.set_color(route_colors[route])
            label.set_fontweight('bold')
            label.set_fontsize(10.0)
    else:
        ax.set_yticklabels([])
panel_c = save_panel(fig, 'fig4c_pinn_parity_dual_error_axes.png')

# %%


# Fig. 4d: mechanism/order table plus fitted activation terms.
fig = plt.figure(figsize=(7.2, 4.33))
panel_label(fig, 'd', 'Kinetic interpretation')
k = kin_params.set_index('route_label').loc[route_order].reset_index()
reaction_rows = [
    (r'$t$-BuCl + H$_2$O$_2$ → TBHP', 'Side step not shown', 'order_H2O2', 'order_TBCL'),
    (r'TBA + H$_2$O$_2$ → TBHP', r'Cat.: WPO$_4$', 'order_H2O2', 'order_TBA_main'),
    (r'TBA + H$_2$O$_2$ → TBHP', r'Cat.: CF$_3$', 'order_H2O2', 'order_TBA_main'),
    (r'BzH + TBHP → TBPB', 'Side step not shown', 'order_benzaldehyde', 'order_TBHP'),
]

table = fig.add_axes([0.075, 0.525, 0.86, 0.405])
table.set(xlim=(0, 1), ylim=(0, 1)); table.set_axis_off()
xcols = [0.02, 0.29, 0.85, 0.96]
for x, label in zip(xcols, ['Route', 'Main reaction', r'$a$', r'$b$']):
    table.text(x, 0.96, label, fontsize=10.5, color='#555555',
               ha='center' if x >= 0.84 else 'left', va='baseline')
for i, (row, meta) in enumerate(zip(k.itertuples(index=False), reaction_rows)):
    reaction, note, acol, bcol = meta
    y = 0.745 - i * 0.200
    table.text(xcols[0], y, row.route_label, fontsize=11.0, weight='bold',
               color=route_colors[row.route_label], va='baseline')
    table.text(xcols[1], y, reaction, fontsize=10.2, color='#222222', va='baseline')
    table.text(xcols[1], y - 0.075, note, fontsize=9.2, color='#444444', va='baseline')
    table.text(xcols[2], y, f'{getattr(row, acol):.2f}', fontsize=10.2, ha='center', va='baseline')
    table.text(xcols[3], y, f'{getattr(row, bcol):.2f}', fontsize=10.2, ha='center', va='baseline')

fig.text(0.075, 0.450, 'Activation terms', fontsize=12.5, fontweight='bold')
fig.text(0.60, 0.450, '● Main   □ Side', fontsize=10.0, color='#333333')
ax_param = fig.add_axes([0.16, 0.095, 0.76, 0.30])
main = k['Ea1_over_R'].to_numpy(float) / 1000
side = k['Ea2_over_R'].to_numpy(float) / 1000
x = np.arange(len(route_order))
for i, route in enumerate(route_order):
    ax_param.vlines(i - 0.10, 0, main[i], color=main_color, lw=1.0, alpha=0.45)
    ax_param.scatter(i - 0.10, main[i], s=30, color=main_color, zorder=3)
    ax_param.annotate(f'{main[i]:.3f}', (i - 0.10, main[i]), xytext=(0, 5),
                      textcoords='offset points', ha='center', va='bottom', fontsize=9.4, color=main_color)
    if np.isfinite(side[i]):
        ax_param.vlines(i + 0.10, 0, side[i], color=side_color, lw=1.0, alpha=0.45)
        ax_param.scatter(i + 0.10, side[i], s=30, marker='s', facecolor='white',
                         edgecolor=side_color, linewidth=1.1, zorder=4)
        ax_param.annotate(f'{side[i]:.3f}', (i + 0.10, side[i]), xytext=(0, -6),
                          textcoords='offset points', ha='center', va='top', fontsize=9.4, color=side_color)
ax_param.set_xticks(x, route_order, rotation=0)
ax_param.set_ylabel(r'$E_{\mathrm{a}}/R$ ($10^3$ K)', fontsize=10.5)
ax_param.set_ylim(0, max(np.nanmax(main), np.nanmax(side)) * 1.25)
finish_axes(ax_param, grid_axis='y')
ax_param.tick_params(labelsize=9.5)
panel_d = save_panel(fig, 'fig4d_mechanism_activation_terms.png')


# %%

# Fig. 4e: real one-factor measurements and model-recommended conditions.
fig = plt.figure(figsize=(7.2, 4.33))
panel_label(fig, 'e', 'Condition optimization')

# The representative route is selected from current 0720 PINN outputs: among TBHP routes (Path 1-3), WPO4 has the highest best_model objective.
route_key = max(['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3'], key=lambda key: condition_cmp[(condition_cmp['route'] == key) & (condition_cmp['kind'] == 'best_model')]['score'].iloc[0])
route_label = {'tbhp_bzcl': 'Route 1', 'tbhp_wpo4': 'Route 2', 'tbhp_cf3': 'Route 3'}[route_key]
df = route_training(route_key)
u = df.groupby('experiment_id').agg({'H2O2_0':'first','TBA_0':'first','Catalyst_0':'first','T_0':'first','TBHP':'max'}).reset_index()
u['ratio'] = u['H2O2_0'] / u['TBA_0']
scans = [
    ('Catalyst eq.', 'Catalyst_0', u[(np.isclose(u['ratio'], 1.5)) & (np.isclose(u['T_0'], 70.0))].sort_values('Catalyst_0')),
    (r'H$_2$O$_2$/TBA', 'ratio', u[(np.isclose(u['Catalyst_0'], 0.116122, rtol=0, atol=1e-4)) & (np.isclose(u['T_0'], 70.0))].sort_values('ratio')),
    ('Temperature (°C)', 'T_0', u[(np.isclose(u['ratio'], 2.2)) & (np.isclose(u['Catalyst_0'], 0.108864, rtol=0, atol=1e-4))].sort_values('T_0')),
]

def scan_optimum_label(xlabel, value):
    if 'Temperature' in xlabel:
        return f'{value:.0f}°C'
    return f'{value:.2f}'

axes = []
for i, (xlabel, xcol, sub) in enumerate(scans):
    ax = fig.add_axes([0.10 + i * 0.30, 0.57, 0.25, 0.25])
    axes.append(ax)
    x = sub[xcol].to_numpy(float)
    y = sub['TBHP'].to_numpy(float)
    ax.plot(x, y, color=route_colors[route_label], lw=1.80, marker='o', ms=4.2)
    ymax = max(3.0, np.nanmax(y) * 1.15)
    best_idx = int(np.nanargmax(y))
    ax.scatter([x[best_idx]], [y[best_idx]], s=28, color='#D8893A', edgecolor='black', linewidth=0.3, zorder=5)
    ax.text(x[best_idx], y[best_idx] + 0.10, scan_optimum_label(xlabel, x[best_idx]), ha='center', va='bottom', fontsize=10.0, color='#222222')
    ax.set_xlabel(xlabel, fontsize=10.4)
    ax.set_ylim(0, ymax)
    finish_axes(ax)
axes[0].set_ylabel('TBHP\n(mol L$^{-1}$)', fontsize=10.5)
fig.text(0.10, 0.875, f"{route_label.replace('Route', 'Path')}: experiments", color=route_colors[route_label], fontsize=12.0, fontweight='bold')
for ax in axes[1:]:
    ax.set_yticklabels([])

model = condition_cmp[condition_cmp['kind'] == 'best_model'].set_index('route_label').loc[route_order]
fig.text(0.065, 0.43, 'PINN optima', fontsize=12.5, fontweight='bold')
table = fig.add_axes([0.065, 0.07, 0.87, 0.31])
table.set(xlim=(0, 1), ylim=(0, 1)); table.set_axis_off()
xcols = [0.02, 0.34, 0.52, 0.71, 0.91]
table.axhspan(0.80, 1.0, color='#F2F5F6', zorder=0)
for j, (xpos, header) in enumerate(zip(xcols, ['Route', 'Ratio', r'$T$ (°C)', 'Catalyst', 'Yield'])):
    table.text(xpos, 0.90, header, ha='left' if j == 0 else 'center', va='center', fontsize=10.0, color='#555555')
table.plot([0, 1], [0.79, 0.79], color='#D9DEE2', lw=0.8)
for i, route in enumerate(route_order):
    row = model.loc[route]
    if route == 'Route 1':
        ratio = row['H2O2'] / row['TBCL']; cat = '/'
    elif route in ['Route 2', 'Route 3']:
        ratio = row['H2O2'] / row['TBA']; cat = f"{row['Catalyst']:.2f}"
    else:
        ratio = row['TBHP'] / row['Benzaldehyde']; cat = f"{row['Catalyst']:.2f}"
    y0 = 0.68 - i * 0.17
    values = [route, f'{ratio:.2f}', f"{row['T']:.0f}", cat, f"{row['yield']:.2f}"]
    for j, (xpos, value) in enumerate(zip(xcols, values)):
        table.text(xpos, y0, value, ha='left' if j == 0 else 'center', va='center',
                   fontsize=10.5, fontweight='bold' if j == 0 else 'normal',
                   color=route_colors[route] if j == 0 else '#222222')
    table.plot([0, 1], [y0 - 0.085, y0 - 0.085], color='#D9DEE2', lw=0.55)
panel_e = save_panel(fig, 'fig4e_lead_route_condition_scans_text_optima.png')

# %%

# Fig. 4f: route scores from the original route-evaluation output.
fig = plt.figure(figsize=(7.2, 4.33))
panel_label(fig, 'f', 'Route evaluation')
metrics = [
    ('Yield', 'yield_score_0_5'),
    ('Selectivity', 'selectivity_score_0_5'),
    ('Productivity', 'productivity_score_0_5'),
    ('Kinetic\naccess.', 'kinetic_accessibility_score_0_5'),
    ('Greenness (GWP)', 'greenness_score_0_5'),
]
scores = route_eval.set_index('route_label').loc[route_order]
mat = scores[[col for _, col in metrics]].to_numpy(float).T
cmap = mpl.colors.LinearSegmentedColormap.from_list('score', ['#F1F5F7', '#B8CDD8', '#52758F'])
ax = fig.add_axes([0.225, 0.235, 0.675, 0.625])
mesh = ax.pcolormesh(np.arange(5), np.arange(6), mat, cmap=cmap, vmin=3, vmax=5,
                     edgecolors='white', linewidth=1.5)
ax.set(xlim=(0, 4), ylim=(5, 0))
ax.set_xticks(np.arange(4) + 0.5, route_order)
ax.xaxis.tick_top()
ax.set_yticks(np.arange(5) + 0.5, [label for label, _ in metrics])
ax.tick_params(length=0, pad=4, labelsize=10.5)
for tick, route in zip(ax.get_xticklabels(), route_order):
    tick.set_color(route_colors[route]); tick.set_fontweight('bold')
for spine in ax.spines.values(): spine.set_visible(False)
for row in range(5):
    for col in range(4):
        value = mat[row, col]
        ax.text(col + 0.5, row + 0.5, f'{value:.1f}', ha='center', va='center', fontsize=12.0,
                color='white' if value >= 4.6 else '#222222')
overall = scores['overall_score_0_5'].to_numpy(float)
summary = fig.add_axes([0.225, 0.115, 0.675, 0.090])
summary.set(xlim=(0, 4), ylim=(0, 1)); summary.set_axis_off()
summary.axhspan(0, 1, color='#F2F5F6')
for i, value in enumerate(overall):
    summary.text(i + 0.5, 0.5, f'{value:.1f}', ha='center', va='center', fontsize=12.0, fontweight='bold')
fig.text(0.205, 0.160, 'Overall', ha='right', va='center', fontsize=10.5, fontweight='bold')
cax = fig.add_axes([0.38, 0.040, 0.40, 0.026])
cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[3, 4, 5])
cb.outline.set_visible(False); cb.ax.tick_params(length=0, pad=2, labelsize=9.5)
fig.text(0.35, 0.053, 'Score', ha='right', va='center', fontsize=9.5, color='#555555')
panel_f = save_panel(fig, 'fig4f_quantitative_route_evaluation.png')

# %%


# Assemble a six-panel composite preview
panel_paths = [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]
images = [Image.open(p).convert('RGB') for p in panel_paths]
target_w, target_h = 1600, 960
tiles = []
for img in images:
    img = ImageOps.contain(img, (target_w - 18, target_h - 18), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (target_w, target_h), 'white')
    canvas.paste(img, ((target_w - img.width) // 2, (target_h - img.height) // 2))
    tiles.append(canvas)
cols, rows = 2, 3
comp = Image.new('RGB', (cols * target_w, rows * target_h), 'white')
for i, tile in enumerate(tiles):
    comp.paste(tile, ((i % cols) * target_w, (i // cols) * target_h))
png_path = COMPOSITE_ROOT / 'Fig4_revised_0727_composite.png'
pdf_path = COMPOSITE_ROOT / 'Fig4_revised_0727_composite.pdf'
comp.save(png_path)
fig, ax = plt.subplots(figsize=(cols * target_w / 300, rows * target_h / 300), dpi=300)
ax.imshow(comp)
ax.set_axis_off()
fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.0, facecolor='white')
plt.close(fig)
display(IPImage(filename=str(png_path)))
png_path, pdf_path
