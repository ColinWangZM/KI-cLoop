from __future__ import annotations
import runtime
import argparse
import fcntl
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from kinetic_core import HERE, ROOT, CONFIGS, bounds, export_predictions, fit, load_data, metrics, physical_to_scaled, predict, scaled_to_physical

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)

class Progress:

    def __init__(self, out, key):
        self.out, self.key = (out, key)
        self.started = time.monotonic()

    def __call__(self, stage, done, total, message=''):
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        value = {'route': self.key, 'stage': stage, 'done': done, 'total': total, 'message': message, 'timestamp': now, 'elapsed_s': time.monotonic() - self.started, 'pid': os.getpid()}
        write_json(self.out / 'progress.json', value)
        print(f'[{now}] [{self.key}] {stage} {done}/{total} {message}', flush=True)

def starts_for(data, count, seed):
    rng = np.random.default_rng(seed)
    cfg = CONFIGS[data.key]
    starts = [(f'fitted_initialization_{i + 1}', physical_to_scaled(data, p)) for i, p in enumerate(cfg['fitted_initializations'])]
    neutral = dict(zip(cfg['param_names'], cfg['initial_params']))
    starts.append(('neutral', physical_to_scaled(data, neutral)))
    lo, hi = bounds(data)
    anchor = starts[0][1]
    while len(starts) < count:
        q = anchor + rng.normal(size=len(anchor)) * np.tile([0.6, 0.12, 0.4, 0.4, 0.4], len(data.reactants))
        if data.key == 'tbhp_bzcl':
            q[2] = 0.0
        starts.append((f'random_{len(starts)}', np.clip(q, lo + 1e-05, hi - 1e-05)))
    return starts[:count]

def baseline(data, args, out, progress):
    target = out / 'baseline.json'
    if args.resume and target.exists():
        saved = json.loads(target.read_text())
        return (np.array(saved['q']), saved['step'])
    records = []
    candidates = starts_for(data, args.starts, args.seed)
    for i, (label, q0) in enumerate(candidates):
        progress('ode_multistart', i, args.starts, label)
        q, info = fit(data, q0, step=args.step, max_nfev=args.max_nfev, callback=lambda n, s: progress('ode_multistart', i, args.starts, f'{label}: evaluations={n}, SSE={s:.6g}'))
        records.append({'start': label, 'q': q.tolist(), 'parameters': scaled_to_physical(data, q), **info})
        write_json(out / 'multistart.json', records)
        progress('ode_multistart', i + 1, args.starts, f"{label}: SSE={info['sse']:.7g}, converged={info['success']}")
    successful = [r for r in records if r['success']]
    if not successful:
        raise RuntimeError('No multistart optimization converged.')
    best = min(successful, key=lambda r: r['sse'])
    q = np.array(best['q'])
    step = args.step
    numerical = []
    for level in range(6):
        full = predict(data, q, step)
        finer = predict(data, q, step / 2)
        gap = float(np.max(abs(full - finer)))
        numerical.append({'step': step, 'half_step_max_gap': gap})
        progress('integration_check', level + 1, 6, f'step={step:g}, half-step gap={gap:.4g}')
        if gap < args.integration_tol:
            break
        step /= 2
        q, info = fit(data, q, step=step, max_nfev=args.max_nfev)
    else:
        raise RuntimeError('Integrator refinement did not meet the requested tolerance.')
    q, info = fit(data, q, step=step, max_nfev=args.max_nfev)
    again, again_info = fit(data, q, step=step, max_nfev=args.max_nfev)
    change = np.max(abs(again - q))
    if again_info['sse'] < info['sse']:
        q, info = (again, again_info)
    result = {'method': 'conservative_ODE_same_model_baseline', 'q': q.tolist(), 'step': step, 'parameters': scaled_to_physical(data, q), 'optimizer': info, 'second_refit_max_scaled_parameter_change': float(change), 'numerical_refinement': numerical, 'metrics': metrics(data, predict(data, q, step)), 'loss': 'unweighted concentration SSE, noninitial points; no parameter prior', 'initialization_policy': 'supplied fitted starts, neutral start and seeded perturbations; minimum converged SSE'}
    write_json(target, result)
    export_predictions(data, predict(data, q, step), out / 'baseline_predictions.csv')
    return (q, step)

def bootstrap(data, q, step, args, out, progress):
    folder = out / 'bootstrap'
    folder.mkdir(exist_ok=True)
    prediction = predict(data, q, step)
    residual = data.observations - prediction
    rng = np.random.default_rng(args.seed + 700)
    multipliers = rng.choice([-1.0, 1.0], size=(args.bootstrap, len(data.times)))
    seeds = rng.integers(0, 2 ** 31 - 1, size=args.bootstrap)
    completed = {}
    for i in range(args.bootstrap):
        p = folder / f'{i:04d}.json'
        if args.resume and p.exists():
            completed[i] = json.loads(p.read_text())

    def one(i):
        target = prediction + multipliers[i, :, None, None] * residual
        learned, info = fit(data, q, target=target, step=step, max_nfev=args.max_nfev)
        retried = False
        if not info['success']:
            jitter = np.random.default_rng(int(seeds[i])).normal(0, 0.01, size=len(q))
            candidate, other = fit(data, q + jitter, target=target, step=step, max_nfev=2 * args.max_nfev)
            retried = True
            if other['success'] or other['sse'] < info['sse']:
                learned, info = (candidate, other)
        value = {'replicate': i, 'q': learned.tolist(), 'parameters': scaled_to_physical(data, learned), 'optimizer': info, 'retried': retried, 'multipliers': multipliers[i].tolist()}
        write_json(folder / f'{i:04d}.json', value)
        return (i, value)
    pending = [i for i in range(args.bootstrap) if i not in completed]
    progress('bootstrap', len(completed), args.bootstrap, f'workers={args.workers}, whole-condition wild residual bootstrap')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one, i) for i in pending]
        for future in concurrent.futures.as_completed(futures):
            i, value = future.result()
            completed[i] = value
            success = sum((v['optimizer']['success'] for v in completed.values()))
            progress('bootstrap', len(completed), args.bootstrap, f"converged={success}, last nfev={value['optimizer']['nfev']}")
    rows = [{'replicate': i, 'converged': v['optimizer']['success'], **v['parameters']} for i, v in sorted(completed.items())]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'bootstrap_parameters.csv', index=False)
    valid = frame[frame.converged].drop(columns=['replicate', 'converged'])
    estimates = scaled_to_physical(data, q)
    intervals = {name: {'estimate': estimates[name], 'low': float(valid[name].quantile(0.025)), 'high': float(valid[name].quantile(0.975)), 'sd': float(valid[name].std())} for name in valid}
    enough = len(valid) >= max(100, 0.95 * args.bootstrap)
    write_json(out / 'uncertainty.json', {'method': 'whole-condition wild residual bootstrap', 'estimator': 'final conservative ODE refinement, conditional on model and observed conditions', 'not_full_neural_network_bootstrap': True, 'requested': args.bootstrap, 'converged': len(valid), 'sufficient_replicate_count': enough, 'seed': args.seed + 700, 'intervals': intervals, 'notes': ['No independent-error assumption across derived species or times in one condition.', 'Intervals do not cover model-form error, initial-concentration error or extrapolation.', 'Reported percentiles are conditional estimates; bound contacts must be inspected.']})

def diagnostics(data, q, step, args, out, progress):
    rows = []
    folds = np.array_split(np.random.default_rng(args.seed + 901).permutation(len(data.times)), args.folds)
    for i, test in enumerate(folds):
        if len(test) == 0:
            continue
        train = np.array([j for j in range(len(data.times)) if j not in test])
        training = data.subset(train)
        neutral = dict(zip(CONFIGS[data.key]['param_names'], CONFIGS[data.key]['initial_params']))
        q0 = physical_to_scaled(training, neutral)
        learned, info = fit(training, q0, step=step, max_nfev=args.max_nfev)
        held = data.subset(test)
        value = {'fold': i, 'held_out_conditions': held.sheets, 'optimizer': info, 'metrics': metrics(held, predict(held, learned, step)), 'parameters': scaled_to_physical(data, learned)}
        rows.append(value)
        progress('condition_cross_validation', i + 1, args.folds, f'held out {len(test)} conditions')
    write_json(out / 'condition_cross_validation.json', rows)
    wide, info = fit(data, q, step=step, max_nfev=args.max_nfev, wide=True)
    write_json(out / 'wide_bounds_sensitivity.json', {'q': wide.tolist(), 'parameters': scaled_to_physical(data, wide), 'optimizer': info, 'metrics': metrics(data, predict(data, wide, step))})
    selected = [1]
    if len(q) > 5:
        selected += [6, 3, 4, 8, 9]
    elif data.key == 'tbpb_benzaldehyde':
        selected += [2, 3]
    low, high = bounds(data)
    records = []
    grids = {}
    for index in selected:
        width = 0.15 if index % 5 == 1 else 0.5
        values = np.linspace(max(low[index] + 1e-07, q[index] - width), min(high[index] - 1e-07, q[index] + width), args.profile_points)
        grids[index] = np.unique(np.r_[values, q[index]])
    total = sum((len(values) for values in grids.values()))
    for index, values in grids.items():
        for value in values:
            learned, info = fit(data, q, step=step, max_nfev=args.max_nfev, fixed=(index, float(value)))
            records.append({'index': index, 'fixed_scaled_value': float(value), **info, 'parameters': scaled_to_physical(data, learned)})
            progress('profile_objective', len(records), total, f"parameter index={index}, SSE={info['sse']:.6g}")
    write_json(out / 'profile_objective.json', {'interpretation': 'reoptimized SSE sensitivity; not a likelihood-ratio CI', 'profiles': records})

def run_route(key, args):
    out = args.output / key
    out.mkdir(parents=True, exist_ok=True)
    progress = Progress(out, key)
    data = load_data(key)
    source_names = ['kinetic_core.py', 'pinn_model.py', 'train.py', 'conservative_solver.cpp', 'routes.json', 'runtime.py']
    manifest = {'source': data.source, 'route': key, 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'code_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names}}
    if args.resume and (out / 'manifest.json').exists():
        old = json.loads((out / 'manifest.json').read_text())
        if old['source']['sha256'] != data.source['sha256']:
            raise RuntimeError('Source data changed; choose a new output directory.')
        if old['code_sha256'] != manifest['code_sha256']:
            raise RuntimeError('Training code changed; choose a new output directory instead of mixing runs.')
        for name in ['seed', 'epochs', 'warmup', 'lbfgs', 'step', 'starts', 'integration_tol']:
            if old['args'][name] != manifest['args'][name]:
                raise RuntimeError(f'{name} changed; choose a new output directory.')
    write_json(out / 'manifest.json', manifest)
    export_predictions(data, data.observations, out / 'input_data.csv')
    progress('data_checked', 1, 1, f'{len(data.times)} conditions, {data.observations.size} species observations')
    q, step = baseline(data, args, out, progress)
    if args.stage == 'baseline':
        progress('baseline_complete', 1, 1)
        return
    if args.stage in {'all', 'pinn'}:
        from pinn_model import train_pinn
        if args.resume and (out / 'pinn_summary.json').exists():
            neural = json.loads((out / 'pinn_summary.json').read_text())
            q_nn = np.array(neural['q'])
        else:
            q_nn, neural, net_prediction = train_pinn(data, q, out, epochs=args.epochs, warmup=args.warmup, lbfgs=args.lbfgs, seed=args.seed, progress=progress, step=step)
            export_predictions(data, net_prediction, out / 'pinn_network_predictions.csv')
        refined, info = fit(data, q_nn, step=step, max_nfev=args.max_nfev)
        baseline_sse = float(np.sum((predict(data, q, step)[:, 1:] - data.observations[:, 1:]) ** 2))
        if info['sse'] > baseline_sse * (1 + 1e-07):
            refined, info = fit(data, q, step=step, max_nfev=args.max_nfev)
        q = refined
        numerical_gap = float(np.max(abs(predict(data, q, step) - predict(data, q, step / 2))))
        result = {'method': 'hybrid_PINN_with_conservative_ODE_likelihood_refinement', 'q': q.tolist(), 'parameters': scaled_to_physical(data, q), 'step': step, 'optimizer': info, 'metrics': metrics(data, predict(data, q, step)), 'half_step_max_gap': numerical_gap, 'same_data_ODE_baseline_used_for_warmup': True, 'parameter_prior_weight': 0.0, 'raw_PINN_parameters_file': 'pinn_summary.json'}
        write_json(out / 'final_summary.json', result)
        export_predictions(data, predict(data, q, step), out / 'final_predictions.csv')
        progress('ode_refinement_complete', 1, 1, f"SSE={info['sse']:.7g}; step={step:g}")
    if args.stage == 'pinn':
        progress('pinn_and_refinement_complete', 1, 1)
        return
    if args.stage == 'stats':
        final = json.loads((out / 'final_summary.json').read_text())
        q = np.array(final['q'])
        step = final['step']
    bootstrap(data, q, step, args, out, progress)
    diagnostics(data, q, step, args, out, progress)
    progress('complete', 1, 1, 'PINN, forward verification, bootstrap, condition validation and profiles complete')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=['all', *CONFIGS], default='all')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs' / 'training')
    parser.add_argument('--stage', choices=['all', 'baseline', 'pinn', 'stats'], default='all')
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=1500)
    parser.add_argument('--lbfgs', type=int, default=600)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--profile-points', type=int, default=7)
    parser.add_argument('--max-nfev', type=int, default=1000)
    parser.add_argument('--step', type=float, default=2.0)
    parser.add_argument('--integration-tol', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.warmup, args.max_nfev, args.workers) < 1 or args.bootstrap < 2 or args.folds < 2 or (args.starts < 2):
        parser.error('Use positive training settings, at least 2 starts/folds/bootstrap samples.')
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = (args.output / '.training.lock').open('a+')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A training process is already using this output directory.')
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['kinetic_core.py', 'pinn_model.py', 'train.py', 'runtime.py', 'routes.json', 'conservative_solver.cpp']:
        destination = snapshot / name
        if not destination.exists():
            destination.write_bytes((HERE / name).read_bytes())
    for key in CONFIGS if args.route == 'all' else [args.route]:
        try:
            run_route(key, args)
        except Exception as error:
            out = args.output / key
            out.mkdir(exist_ok=True)
            Progress(out, key)('failed', 0, 1, str(error))
            raise
if __name__ == '__main__':
    main()
