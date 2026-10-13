import runtime
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from kinetic_core import HERE, CONFIGS, load_data, predict, metrics, scaled_to_physical

def verify(directory):
    report = []
    for key, cfg in CONFIGS.items():
        data = load_data(key)
        result = json.loads((directory / key / 'final_summary.json').read_text())
        q = np.array(result['q'])
        prediction = predict(data, q, result['step'])
        reference = pd.read_csv(directory / key / 'final_predictions.csv', float_precision='round_trip')
        expected = reference[[s + '_pred' for s in data.species]].to_numpy().reshape(prediction.shape)
        np.testing.assert_allclose(prediction, expected, atol=2e-09, rtol=2e-09)
        measured = reference[data.species].to_numpy().reshape(data.observations.shape)
        np.testing.assert_allclose(data.observations, measured, atol=1e-13, rtol=1e-13)
        recomputed = metrics(data, prediction)
        for name, value in scaled_to_physical(data, q).items():
            np.testing.assert_allclose(value, result['parameters'][name], atol=2e-10, rtol=2e-10)
        for species in data.species:
            for measure in ['r2', 'rmse', 'bias']:
                np.testing.assert_allclose(recomputed[species + '_noninitial'][measure], result['metrics'][species + '_noninitial'][measure], atol=1e-10, rtol=1e-10)
        draws = pd.read_csv(directory / key / 'bootstrap_parameters.csv')
        valid = draws[draws['converged']]
        uncertainty = json.loads((directory / key / 'uncertainty.json').read_text())
        assert len(draws) == uncertainty['requested'] and len(valid) == uncertainty['converged']
        for name, values in uncertainty['intervals'].items():
            np.testing.assert_allclose(valid[name].quantile([0.025, 0.975]), [values['low'], values['high']], atol=1e-09, rtol=1e-09)
        target = 'TBPB' if key == 'tbpb_benzaldehyde' else 'TBHP'
        report.append({'route': key, 'conditions': len(data.times), 'states': int(data.times.size), 'noninitial_states': int(data.times[:, 1:].size), 'Ea1_kJ_mol': result['parameters']['Ea1_kJ_mol'], 'product_noninitial_R2': recomputed[target + '_noninitial']['r2'], 'prediction_max_abs_error': float(np.max(abs(prediction - expected))), 'bootstrap_converged': len(valid), 'max_invariant_error': recomputed['physical_checks']['max_invariant_error']})
    assert sum((row['conditions'] for row in report)) == 60
    assert sum((row['states'] for row in report)) == 420
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify supplied predictions, metrics and bootstrap intervals.')
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.results)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
