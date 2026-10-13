import runtime
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent

def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=HERE, check=True)

def main():
    parser = argparse.ArgumentParser(description='Reproduce the hybrid PINN-ODE kinetic analysis.')
    parser.add_argument('--mode', choices=['check', 'baseline', 'smoke', 'primary', 'full'], default='check')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from build_solver import build
    build()
    if args.mode == 'check':
        run(HERE / 'test_physics.py')
        run(HERE / 'verify_results.py')
        return
    output = (args.output or HERE / 'outputs' / args.mode).resolve()
    reference = HERE / 'results'
    if output == reference or reference in output.parents:
        parser.error('Choose a generated-output directory outside results/.')
    if output.exists() and any(output.iterdir()) and (not args.resume):
        parser.error('Output is not empty; choose a new directory or use --resume.')
    output.mkdir(parents=True, exist_ok=True)
    packages = {name: metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'torch']}
    (output / 'environment.json').write_text(json.dumps({'python': sys.version, 'packages': packages, 'mode': args.mode}, indent=2) + '\n')
    settings = ['--route', 'all', '--output', output, '--seed', '20260917']
    if args.resume:
        settings += ['--resume']
    if args.mode == 'baseline':
        settings += ['--stage', 'baseline', '--starts', '8']
    elif args.mode == 'smoke':
        settings += ['--stage', 'pinn', '--starts', '2', '--epochs', '20', '--warmup', '10', '--lbfgs', '5']
    else:
        settings += ['--starts', '8', '--epochs', '10000', '--warmup', '1500', '--lbfgs', '600', '--bootstrap', '200', '--folds', '5', '--workers', '8', '--profile-points', '7']
    run(HERE / 'train.py', *settings)
    if args.mode in {'primary', 'full'}:
        run(HERE / 'check_parameter_sensitivity.py', '--output', output)
    if args.mode == 'full':
        run(HERE / 'check_neural_seeds.py', '--output', output)
        run(HERE / 'neutral_initialization_sensitivity.py', '--output', output)
    if args.mode == 'smoke':
        print('Short execution check only; publication settings are provided by primary/full.')
if __name__ == '__main__':
    main()
