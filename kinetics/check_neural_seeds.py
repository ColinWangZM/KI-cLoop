import runtime
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from kinetic_core import HERE, CONFIGS, load_data, fit, scaled_to_physical
from pinn_model import train_pinn
from train import Progress, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'training')
    parser.add_argument('--seeds', nargs='+', type=int, default=[20260918, 20260919])
    args = parser.parse_args()
    summaries, parameters = ([], [])
    for key in CONFIGS:
        directory = args.output / key
        baseline = json.loads((directory / 'baseline.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        data = load_data(key)
        if data.source['sha256'] != manifest['source']['sha256']:
            raise RuntimeError('Input data changed; seed checks require matching inputs.')
        for name, digest in manifest['code_sha256'].items():
            if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Training source changed: {name}')
        settings = manifest['args']
        for seed in [settings['seed'], *args.seeds]:
            if seed == settings['seed']:
                neural = json.loads((directory / 'pinn_summary.json').read_text())
                final = json.loads((directory / 'final_summary.json').read_text())
            else:
                target = directory / 'neural_seeds' / str(seed)
                target.mkdir(parents=True, exist_ok=True)
                progress = Progress(target, key + ':seed=' + str(seed))
                expected = {'seed': seed, 'training_code_sha256': manifest['code_sha256'], 'source_sha256': data.source['sha256'], 'settings': settings}
                provenance = target / 'provenance.json'
                if provenance.exists() and json.loads(provenance.read_text()) != expected:
                    raise RuntimeError('Seed run provenance changed; use another output directory.')
                write_json(provenance, expected)
                if (target / 'refinement.json').exists():
                    neural = json.loads((target / 'pinn_summary.json').read_text())
                    final = json.loads((target / 'refinement.json').read_text())
                else:
                    q, neural, _ = train_pinn(data, np.array(baseline['q']), target, epochs=settings['epochs'], warmup=settings['warmup'], lbfgs=settings['lbfgs'], seed=seed, progress=progress, step=baseline['step'])
                    refined, info = fit(data, q, step=baseline['step'], max_nfev=settings['max_nfev'])
                    final = {'parameters': scaled_to_physical(data, refined), 'optimizer': info}
                    write_json(target / 'refinement.json', final)
                    progress('complete', 1, 1, 'full neural repeat and independent ODE refinement complete')
            summaries.append({'route': key, 'seed': seed, 'adam_epochs': neural['adam_epochs_executed'], 'network_ode_max_gap': neural['max_network_ode_gap'], 'data_loss': neural['data_loss'], 'physics_loss': neural['physics_loss'], 'refined_sse': final['optimizer']['sse'], 'refinement_converged': final['optimizer']['success']})
            for name, value in neural['parameters'].items():
                parameters.append({'route': key, 'seed': seed, 'parameter': name, 'raw_PINN': value, 'ODE_refined': final['parameters'][name]})
            pd.DataFrame(summaries).to_csv(args.output / 'neural_seed_summary.csv', index=False)
            pd.DataFrame(parameters).to_csv(args.output / 'neural_seed_parameters.csv', index=False)
    print('Completed full neural seed checks. Seed ranges are not statistical confidence intervals.', flush=True)
if __name__ == '__main__':
    main()
