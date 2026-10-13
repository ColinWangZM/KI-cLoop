import argparse
import json
from pathlib import Path
import runtime
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from kinetic_core import Dataset, fit, predict, metrics, scaled_to_physical, export_predictions
from prepare_data import check as check_data

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load():
    frame = pd.read_csv(ROOT / 'data/observations.csv')
    observations, times, conditions, names = [], [], [], []
    for _, group in frame.groupby('experiment_id', sort=False):
        group = group.sort_values('state_index')
        if len(group) != frame.groupby('experiment_id').size().iloc[0] or group.iloc[0].residence_time_s != 0:
            raise ValueError('Every condition must begin at the inlet and use a consistent sampling count.')
        observations.append(group[['HXD_M', 'EA_derived_M']].to_numpy(float))
        times.append(group.residence_time_s.to_numpy(float))
        if np.ptp(group.temperature_C) != 0:
            raise ValueError('Temperature must be constant within a condition.')
        conditions.append([group.iloc[0].temperature_C + 273.15, 1.])
        names.append(group.iloc[0].condition)
    y, t, env = (np.asarray(v, dtype=np.float64) for v in (observations, times, conditions))
    if not (np.isfinite(y).all() and np.isfinite(t).all() and (y >= 0).all() and (np.diff(t) > 0).all()):
        raise ValueError('Invalid observations or residence times.')
    if np.max(abs((y[:, :, 0] - y[:, :, 1]) - (y[:, 0, 0] - y[:, 0, 1])[:, None])) > 1e-12:
        raise ValueError('The stoichiometric invariant is inconsistent.')
    return Dataset('paal_knorr', ['HXD', 'EA_derived'], y, t, env, y[:, 0].copy(),
                   np.array([[-1.], [-1.]]), np.array([[0, 1]], dtype=np.int64),
                   np.array([[1., .5, .5]]), 1 / np.mean(1 / env[:, 0]), np.array([0, 1, 3, 4]), None,
                   {'data': 'data/observations.csv', 'concentration_unit': 'mol/L', 'time_unit': 's',
                    'independent_noninitial_observations': 180,
                    'dependent_species': 'EA is derived stoichiometrically from measured HXD depletion.'}, names)


def neutral(data):
    return np.array([np.log(.001), .1, 0., 1., 1.])


def params(data, q):
    values = scaled_to_physical(data, q)
    values.pop('cat_order_1', None)
    return values


def conversion_metrics(data, prediction):
    observed = 100. * (1. - data.observations[:, 1:, 0] / data.initial[:, :1])
    predicted = 100. * (1. - prediction[:, 1:, 0] / data.initial[:, :1])
    errors = (predicted - observed).ravel()
    return {'r2': float(1 - errors @ errors / np.sum((observed - observed.mean()) ** 2)),
            'rmse_percentage_points': float(np.sqrt(np.mean(errors ** 2))), 'observations': int(observed.size)}


def solver_check(data, q):
    prediction = predict(data, q, .5)
    finer = predict(data, q, .25)
    checks = []
    for i in range(len(data.times)):
        def rhs(t, concentrations):
            ca, cb = np.maximum(concentrations, 0)
            logk = q[0] - q[1] * 1e5 / 8.314 * (1 / data.conditions[i, 0] - 1 / data.tref)
            rate = np.exp(logk) * (ca / .5) ** q[3] * (cb / .5) ** q[4]
            return [-rate, -rate]
        solution = solve_ivp(rhs, (0, data.times[i, -1]), data.initial[i], t_eval=data.times[i],
                             method='DOP853', rtol=1e-11, atol=1e-13)
        if not solution.success:
            raise RuntimeError(solution.message)
        checks.append(float(np.max(abs(solution.y.T - prediction[i]))))
    if max(checks) >= 1e-5:
        raise AssertionError('Conservative solver disagrees with DOP853.')
    return {'max_half_step_gap_M': float(np.max(abs(prediction - finer))), 'max_DOP853_gap_M': max(checks)}


def baseline(data, out):
    rng = np.random.default_rng(20261009)
    starts = [neutral(data)]
    for _ in range(7):
        starts.append(np.array([rng.uniform(-8, -5), rng.uniform(.03, .35), 0., rng.uniform(.4, 2), rng.uniform(.4, 2)]))
    records = []
    for index, start in enumerate(starts):
        q, info = fit(data, start, step=.5, max_nfev=1000)
        records.append({'start': index, 'q': q.tolist(), 'parameters': params(data, q), **info})
        print('ODE start', index + 1, 'SSE', info['sse'], flush=True)
    save(out / 'multistart.json', records)
    successful = [r for r in records if r['success']]
    if not successful:
        raise RuntimeError('No ODE initialization converged.')
    selected = min(successful, key=lambda r: r['sse'])
    q = np.array(selected['q'])
    prediction = predict(data, q, .5)
    save(out / 'baseline.json', {'q': q.tolist(), 'parameters': params(data, q), 'metrics': metrics(data, prediction),
         'conversion_metrics': conversion_metrics(data, prediction), 'method': 'ODE SSE fit selected among eight starts.',
         'independent_noninitial_observations': 180, **solver_check(data, q)})
    export_predictions(data, prediction, out / 'baseline_predictions.csv')
    return q


def cross_validation(data, out):
    rows, combined = [], {}
    for group_type, groups in [('temperature', data.conditions[:, 0]), ('ratio', data.initial[:, 0])]:
        predictions = np.full_like(data.observations, np.nan)
        for value in np.unique(groups):
            test = np.flatnonzero(groups == value)
            train_data = data.subset(np.flatnonzero(groups != value))
            learned, info = fit(train_data, neutral(train_data), step=.5, max_nfev=1000)
            if not info['success']:
                raise RuntimeError(f'{group_type} fold {value} did not converge.')
            held = data.subset(test)
            prediction = predict(held, learned, .5)
            predictions[test] = prediction
            rows.append({'group_type': group_type, 'value': float(value), 'held_out_conditions': held.sheets,
                         'parameters': params(train_data, learned), 'optimizer': info, 'metrics': metrics(held, prediction),
                         'conversion_metrics': conversion_metrics(held, prediction), 'estimator': 'ODE only'})
        combined[group_type] = {'concentration_metrics': metrics(data, predictions),
                                'conversion_metrics': conversion_metrics(data, predictions)}
        export_predictions(data, predictions, out / f'leave_{group_type}_out_predictions.csv')
    save(out / 'group_cross_validation.json', {'folds': rows, 'pooled_metrics': combined,
         'scope': 'ODE estimator cross-validation; complete temperature or feed-ratio groups are held out.'})


def bootstrap(data, q, out, replicates):
    rng = np.random.default_rng(20261010)
    prediction = predict(data, q, .5)
    residual = data.observations - prediction
    rows = []
    for index in range(replicates):
        multiplier = rng.choice([-1., 1.], size=(len(data.times), 1, 1))
        learned, info = fit(data, q, target=prediction + multiplier * residual, step=.5, max_nfev=1000)
        rows.append({'replicate': index, 'converged': info['success'], **params(data, learned)})
        if (index + 1) % 25 == 0:
            print('Bootstrap', index + 1, replicates, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'bootstrap_parameters.csv', index=False)
    valid = frame[frame.converged]
    save(out / 'uncertainty.json', {'estimator': 'ODE refinement, whole-condition wild residual bootstrap',
         'replicates': replicates, 'converged': len(valid), 'not_full_neural_bootstrap': True,
         'excludes': 'Measurement error in fixed experimental inputs and model misspecification.',
         'intervals': {name: {'estimate': value, 'low': float(valid[name].quantile(.025)),
                            'high': float(valid[name].quantile(.975))} for name, value in params(data, q).items()}})


def train(data, q, seed, out, warmup=1500, epochs=10000, lbfgs=600):
    from pinn_model import train_pinn
    destination = out / f'pinn_seed_{seed}'
    destination.mkdir(exist_ok=True)
    learned, summary, network_prediction = train_pinn(data, q, destination, warmup=warmup, epochs=epochs,
        lbfgs=lbfgs, seed=seed, log_every=max(1, min(500, epochs)), step=.5,
        progress=lambda stage, done, total, msg='': print(seed, stage, done, total, msg, flush=True))
    refined, info = fit(data, learned, step=.5, max_nfev=1000)
    if not info['success']:
        raise RuntimeError('Final ODE refinement did not converge.')
    prediction = predict(data, refined, .5)
    save(destination / 'comparison.json', {'seed': seed,
        'raw_PINN': {'q': learned.tolist(), 'parameters': params(data, learned),
                     'network_metrics': metrics(data, network_prediction),
                     'forward_ODE_metrics': metrics(data, predict(data, learned, .5)),
                     'network_ODE_gap_M': summary['max_network_ode_gap']},
        'ODE_refined': {'q': refined.tolist(), 'parameters': params(data, refined), 'optimizer': info,
                        'metrics': metrics(data, prediction), 'conversion_metrics': conversion_metrics(data, prediction)},
        'scope': 'Hybrid PINN-ODE model seed repeatability on the same experimental dataset.'})
    export_predictions(data, network_prediction, destination / 'network_predictions.csv')
    export_predictions(data, predict(data, learned, .5), destination / 'raw_pinn_ode_predictions.csv')
    export_predictions(data, prediction, destination / 'refined_predictions.csv')


def verify(data):
    result = check_data()
    reference = json.loads((ROOT / 'reference_results/seed_comparisons.json').read_text())['20261009']['ODE_refined']
    q = np.array(reference['q'])
    prediction = predict(data, q, .5)
    supplied = pd.read_csv(ROOT / 'reference_results/predictions.csv')
    np.testing.assert_allclose(prediction[:, :, 0].ravel(), supplied.final_ODE_HXD_M, rtol=1e-11, atol=1e-11)
    observed = data.observations[:, :, 0].ravel()
    np.testing.assert_allclose(observed, supplied.observed_HXD_M, rtol=1e-12, atol=1e-12)
    found = metrics(data, prediction)
    np.testing.assert_allclose(found['HXD_noninitial']['r2'], reference['metrics']['HXD_noninitial']['r2'], atol=1e-12)
    result.update({'parameters': params(data, q), 'metrics': found, 'conversion_metrics': conversion_metrics(data, prediction),
                   'solver_checks': solver_check(data, q)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['verify', 'baseline', 'train', 'diagnostics', 'all', 'smoke'], default='verify')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261009, 20261010, 20261011])
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    data = load()
    if args.mode == 'verify':
        print(json.dumps(verify(data), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in {'baseline', 'all'}:
        q = baseline(data, args.output)
    else:
        path = args.output / 'baseline.json'
        if not path.exists():
            path = ROOT / 'reference_results/baseline.json'
        q = np.array(json.loads(path.read_text())['q'])
    if args.mode in {'diagnostics', 'all'}:
        if args.bootstrap < 1:
            raise ValueError('Bootstrap replicates must be positive.')
        cross_validation(data, args.output)
        bootstrap(data, q, args.output, args.bootstrap)
    if args.mode in {'train', 'all'}:
        for seed in args.seeds:
            train(data, q, seed, args.output)
    if args.mode == 'smoke':
        print(json.dumps(verify(data), indent=2))
        train(data, q, args.seeds[0], args.output, warmup=2, epochs=2, lbfgs=2)


if __name__ == '__main__':
    main()
