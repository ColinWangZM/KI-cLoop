import runtime
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from kinetic_core import HERE, load_data, fit, scaled_to_physical

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'training')
    args = parser.parse_args()
    rows = []
    cases = [('tbhp_bzcl', 1, [0.0, 0.1, 0.2, 0.4]), ('tbpb_benzaldehyde', 1, [0.0, 0.1, 0.2, 0.4]), ('tbhp_cf3', 8, [0.005, 0.05, 0.1, 0.2])]
    for key, index, values in cases:
        data = load_data(key)
        final = json.loads((args.output / key / 'final_summary.json').read_text())
        for value in values:
            q, info = fit(data, np.array(final['q']), step=final['step'], fixed=(index, value), max_nfev=1000)
            row = {'route': key, 'fixed_parameter': 'Ea1_kJ_mol' if index == 1 else 'order_TBA_side', 'fixed_value': value * 100 if index == 1 else value, 'sse': info['sse'], 'primary_sse': final['optimizer']['sse'], 'sse_change_percent': 100 * (info['sse'] / final['optimizer']['sse'] - 1), 'optimizer_converged': info['success'], **scaled_to_physical(data, q)}
            rows.append(row)
            print(f"{key}: {row['fixed_parameter']}={row['fixed_value']:g}; SSE change {row['sse_change_percent']:+.3f}%; converged={info['success']}", flush=True)
    pd.DataFrame(rows).to_csv(args.output / 'parameter_constraint_sensitivity.csv', index=False)
if __name__ == '__main__':
    main()
