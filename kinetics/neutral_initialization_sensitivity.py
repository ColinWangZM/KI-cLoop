import runtime
import json
import numpy as np
import pandas as pd
from kinetic_core import HERE, CONFIGS, load_data, physical_to_scaled, scaled_to_physical, bounds, fit
import argparse
parser = argparse.ArgumentParser(description='Assess neutral-centered initialization sensitivity.')
parser.add_argument('--output', type=__import__('pathlib').Path, default=HERE / 'outputs' / 'training')
args = parser.parse_args()
out = args.output
records = []
for key, cfg in CONFIGS.items():
    d = load_data(key)
    final = json.loads((out / key / 'final_summary.json').read_text())
    q0 = physical_to_scaled(d, dict(zip(cfg['param_names'], cfg['initial_params'])))
    rng = np.random.default_rng(20260917)
    lo, hi = bounds(d)
    starts = [q0]
    for i in range(7):
        q = q0 + rng.normal(size=len(q0)) * np.tile([0.6, 0.12, 0.4, 0.4, 0.4], len(d.reactants))
        if key == 'tbhp_bzcl':
            q[2] = 0.0
        starts.append(np.clip(q, lo + 1e-05, hi - 1e-05))
    values = []
    for i, start in enumerate(starts):
        learned, info = fit(d, start, step=final['step'], max_nfev=1000)
        values.append({'start': i, 'initialization': 'neutral' if i == 0 else 'neutral-centered random', 'optimizer': info, 'parameters': scaled_to_physical(d, learned), 'q': learned.tolist()})
        print(key, i, info['success'], info['sse'], flush=True)
    best = min((v for v in values if v['optimizer']['success']), key=lambda v: v['optimizer']['sse'])
    ref = np.array(final['q'])
    difference = float(np.max(abs(np.array(best['q'])[d.free] - ref[d.free])))
    summary = {'purpose': 'Independent sensitivity only; no sensitivity result replaces or selects the reported primary fit.', 'source_of_initial_values': 'Generic neutral values and fixed-seed neutral-centered perturbations only.', 'primary_sse': final['optimizer']['sse'], 'best_neutral_sse': best['optimizer']['sse'], 'sse_difference': best['optimizer']['sse'] - final['optimizer']['sse'], 'max_scaled_free_parameter_difference': difference, 'starts': values}
    (out / key / 'neutral_initialization_sensitivity.json').write_text(json.dumps(summary, indent=2) + '\n')
    records.append(dict(route=key, primary_sse=summary['primary_sse'], best_neutral_sse=summary['best_neutral_sse'], sse_difference=summary['sse_difference'], max_scaled_free_parameter_difference=difference, converged_starts=sum((v['optimizer']['success'] for v in values)), total_starts=len(values)))
pd.DataFrame(records).to_csv(out / 'neutral_initialization_sensitivity.csv', index=False)
