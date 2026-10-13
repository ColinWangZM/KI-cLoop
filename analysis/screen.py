import argparse
import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'kinetics'))
from kinetic_core import load_data, predict, _rates, physical_to_scaled
ORDER = ['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3', 'tbpb_benzaldehyde']
META = {key: {'label': f'Path {i + 1}', 'product': 'TBPB' if i == 3 else 'TBHP'} for i, key in enumerate(ORDER)}

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def objectives(d, initial, final):
    initial = np.broadcast_to(initial, final.shape)
    a, b, p = (final[..., 0], final[..., 1], final[..., 2])
    a0, b0 = (initial[..., 0], initial[..., 1])
    if d.key == 'tbpb_benzaldehyde':
        limit = np.minimum(a0, b0 / 2)
        conversion = (a0 - a) / a0
        denominator = a0 - a
    elif len(d.reactants) == 2:
        limit = np.minimum(a0, b0)
        conversion = (a0 - a) / a0
        denominator = p + final[..., 3]
    else:
        limit = np.minimum(a0, b0)
        conversion = (b0 - b) / b0
        denominator = a0 - a
    y = p / limit
    selectivity = np.divide(p, denominator, out=np.zeros_like(p), where=denominator > 1e-12)
    if d.key == 'tbpb_benzaldehyde':
        score = 0.7 * y + 0.2 * selectivity + 0.1 * conversion - 0.05 * b / b0
    elif len(d.reactants) == 2:
        score = 0.65 * y + 0.25 * selectivity + 0.1 * conversion
    else:
        score = 0.75 * y + 0.2 * selectivity + 0.05 * conversion
    return {'score': score, 'yield': y, 'selectivity': selectivity, 'conversion': conversion, 'product': p, 'limiting_initial': limit}

def feed_ratio(key, y):
    return y[..., 1] / y[..., 0] if key == 'tbpb_benzaldehyde' else y[..., 0] / y[..., 1]

def feasible_candidates(d):
    catalyzed = d.key != 'tbhp_bzcl'
    x = np.c_[d.initial[:, :2], d.conditions[:, 1]] if catalyzed else d.initial[:, :2]
    weights = np.linalg.lstsq(x, np.ones(len(x)), rcond=None)[0]
    closure_error = float(np.max(abs(x @ weights - 1)))
    if closure_error > 1e-10 or (weights <= 0).any():
        raise RuntimeError('Observed recipes do not support the assumed positive mixing constraint.')
    ratios = np.unique(np.round(feed_ratio(d.key, d.initial), 8))
    catalysts = np.unique(d.conditions[:, 1]) if catalyzed else [1.0]
    temperatures = np.arange(d.conditions[:, 0].min() - 273.15, d.conditions[:, 0].max() - 273.15 + 0.001, 1.0)
    rows = []
    for ratio, cat, temp in itertools.product(ratios, catalysts, temperatures):
        available = 1 - weights[2] * cat if catalyzed else 1.0
        if d.key == 'tbpb_benzaldehyde':
            a = available / (weights[0] + weights[1] * ratio)
            b = ratio * a
        else:
            b = available / (weights[0] * ratio + weights[1])
            a = ratio * b
        if min(a, b) <= 0 or any((v < d.initial[:, i].min() - 1e-07 or v > d.initial[:, i].max() + 1e-07 for i, v in enumerate([a, b]))):
            continue
        rows.append({'feed_ratio': ratio, 'catalyst': cat, 'temperature_C': temp, d.species[0] + '_0': a, d.species[1] + '_0': b})
    frame = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    frame.insert(0, 'candidate_id', np.arange(len(frame)))
    audit = {'mixing_species': d.species[:2] + (['Catalyst'] if catalyzed else []), 'mixing_weights_L_per_mol': weights.tolist(), 'observed_max_mixing_closure_error': closure_error, 'feed_ratio_levels': ratios.tolist(), 'catalyst_levels': list(map(float, catalysts)), 'temperature_grid_C': temperatures.tolist(), 'candidate_count_without_time': len(frame), 'scope': 'observed feed-ratio and catalyst levels; 1 C temperature grid; coupled feed mixing; marginal concentrations remain inside measured ranges', 'not_experimentally_validated_combinations': True}
    return (frame, audit)

def candidate_dataset(d, frame, times):
    initial = np.zeros((len(frame), len(d.species)))
    initial[:, :2] = frame[[n + '_0' for n in d.species[:2]]]
    env = np.c_[frame.temperature_C.to_numpy() + 273.15, frame.catalyst.to_numpy()]
    return replace(d, initial=initial, conditions=env, times=np.broadcast_to(times, (len(frame), len(times))).copy())

def independent_at(d, q, t):
    env = d.conditions[0]
    logk = np.array([q[5 * j] - q[5 * j + 1] * 100000 / 8.314 * (1 / env[0] - 1 / d.tref) + q[5 * j + 2] * np.log(env[1] / d.references[j, 0]) for j in range(len(d.reactants))])
    solution = solve_ivp(lambda _, y: d.stoich @ _rates(y, logk, q, d.reactants, d.references), (0, t), d.initial[0], t_eval=[t], method='DOP853', rtol=1e-09, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]

def screen(d, final, out, results):
    q = np.array(final['q'])
    step = min(final['step'], 0.5)
    candidates, audit = feasible_candidates(d)
    candidates.to_csv(out / f'{d.key}_screening_candidates.csv', index=False)
    times = np.linspace(0, float(d.times[:, -1].max()), 241)
    ranked = []
    for start in range(0, len(candidates), 256):
        frame = candidates.iloc[start:start + 256]
        cd = candidate_dataset(d, frame, times)
        pred = predict(cd, q, step)
        if pred.min() < -1e-09:
            raise RuntimeError('Negative screening prediction.')
        obj = objectives(d, cd.initial[:, None, :], pred)
        if obj['yield'].max() > 1 + 1e-08:
            raise RuntimeError('Predicted yield exceeds stoichiometric limit; no clipping allowed.')
        index = np.argmax(obj['score'][:, 1:], axis=1) + 1
        batch = frame.copy()
        batch['time'] = times[index]
        for name, values in obj.items():
            batch[name] = values[np.arange(len(frame)), index]
        for i, name in enumerate(d.species):
            batch[name + '_final'] = pred[np.arange(len(frame)), index, i]
        ranked.append(batch)
        print(f'{d.key}: condition screening {min(start + len(frame), len(candidates))}/{len(candidates)}', flush=True)
    ranked = pd.concat(ranked).sort_values(['score', 'time', 'candidate_id'], ascending=[False, True, True]).reset_index(drop=True)
    n_refine = max(min(40, len(ranked)), int((ranked.score >= ranked.score.max() - 0.005).sum()))
    refined = []
    for i in range(n_refine):
        row = ranked.iloc[i].copy()
        one = pd.DataFrame([row])

        def evaluate(t):
            cd = candidate_dataset(d, one, np.array([0.0, float(t)]))
            pred = predict(cd, q, step / 2)[0, -1]
            obj = objectives(d, cd.initial[0], pred)
            return (pred, {k: float(v) for k, v in obj.items()})
        delta = times[1] - times[0]
        low = max(1e-06, row.time - 2 * delta)
        high = min(times[-1], row.time + 2 * delta)
        opt = minimize_scalar(lambda t: -evaluate(t)[1]['score'], bounds=(low, high), method='bounded', options={'xatol': 0.005})
        tests = [evaluate(float(t)) + (float(t),) for t in [row.time, opt.x, high, low]]
        y, obj, t = max(tests, key=lambda item: item[1]['score'])
        row['time'] = t
        for k, v in obj.items():
            row[k] = v
        for j, name in enumerate(d.species):
            row[name + '_final'] = float(y[j])
        refined.append(row)
    refined = pd.DataFrame(refined).sort_values(['score', 'time', 'candidate_id'], ascending=[False, True, True])
    ranked.to_csv(out / f'{d.key}_screening_grid_rank.csv', index=False)
    refined.to_csv(out / f'{d.key}_screening_refined_rank.csv', index=False)
    best = refined.iloc[0].to_dict()
    best['productivity_mol_l_s'] = best['product'] / best['time']
    selected = candidate_dataset(d, pd.DataFrame([best]), np.array([0.0, best['time']]))
    ys = predict(selected, q, step / 2)[0, -1]
    yi = independent_at(selected, q, best['time'])
    gap = float(np.max(abs(ys - yi)))
    if gap > 0.0003:
        raise RuntimeError(f'{d.key}: selected optimum fails independent integration check: {gap}')
    audit.update(time_grid_points=len(times), time_refined_candidates=n_refine, integration_step=step, refined_step=step / 2, independent_optimum_max_gap=gap, selection_claim='best found within the explicitly recorded finite design grid')
    samples = []
    for _, sample in pd.read_csv(results / d.key / 'bootstrap_parameters.csv').iterrows():
        if not sample['converged']:
            continue
        result = predict(selected, physical_to_scaled(d, sample.to_dict()), step / 2)[0, -1]
        obj = objectives(d, selected.initial[0], result)
        samples.append({'replicate': sample['replicate'], **{k: float(v) for k, v in obj.items()}, 'productivity_mol_l_s': float(obj['product'] / best['time'])})
    samples = pd.DataFrame(samples)
    samples.to_csv(out / f'{d.key}_optimum_bootstrap_predictions.csv', index=False)
    audit['uncertainty_scope'] = '200 parameter bootstrap predictions at the selected condition; not a bootstrap of the full optimization or a measurement-noise prediction interval'
    audit['selected_condition_intervals'] = {name: dict(zip(['low', 'high'], map(float, samples[name].quantile([0.025, 0.975])))) for name in ['yield', 'selectivity', 'product', 'score', 'productivity_mol_l_s']}
    return (best, audit)

def main():
    parser = argparse.ArgumentParser(description='Screen feasible reaction conditions using the hybrid PINN-ODE estimates.')
    parser.add_argument('--results', type=Path, default=ROOT / 'kinetics/results')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/screening')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    optima = []
    measured = []
    audits = {}
    for key in ORDER:
        d = load_data(key)
        final = json.loads((args.results / key / 'final_summary.json').read_text())
        best, audit = screen(d, final, out, args.results)
        optima.append({'route': key, 'route_label': META[key]['label'], **best})
        frame = pd.read_csv(args.results / key / 'final_predictions.csv')
        obj = objectives(d, d.initial[:, None, :], d.observations)
        for name, values in obj.items():
            frame[name] = values.ravel()
        frame['route'] = key
        measured.append(frame)
        audit['measured_peak_product_condition'] = frame.loc[frame[META[key]['product']].idxmax(), ['condition', 'time', META[key]['product']]].to_dict()
        audits[key] = audit
    pd.DataFrame(optima).to_csv(out / 'screened_optimal_conditions.csv', index=False)
    pd.concat(measured).to_csv(out / 'measured_candidate_pool.csv', index=False)
    write_json(out / 'optimization_audit.json', audits)
    print(pd.DataFrame(optima)[['route_label', 'temperature_C', 'time', 'yield']].to_string(index=False))
if __name__ == '__main__':
    main()
