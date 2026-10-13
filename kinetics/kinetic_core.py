from __future__ import annotations
import runtime
import hashlib
import ctypes
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIGS = json.loads((HERE / 'routes.json').read_text())
R = 8.314

@dataclass
class Dataset:
    key: str
    species: list
    observations: np.ndarray
    times: np.ndarray
    conditions: np.ndarray
    initial: np.ndarray
    stoich: np.ndarray
    reactants: np.ndarray
    references: np.ndarray
    tref: float
    free: np.ndarray
    initial_parameters: dict | None
    source: dict
    sheets: list

    def subset(self, indices):
        indices = np.asarray(indices, dtype=int)
        return replace(self, observations=self.observations[indices], times=self.times[indices], conditions=self.conditions[indices], initial=self.initial[indices], sheets=[self.sheets[i] for i in indices])

def load_data(key):
    cfg = CONFIGS[key]
    path = HERE / cfg['data_file']
    frame = pd.read_csv(path, float_precision='round_trip')
    groups = list(frame.groupby('experiment_id', sort=False))
    species = cfg['fitted_species']
    names = [str(g['condition'].iloc[0]) for _, g in groups]
    if names != cfg['sheet_list'] or any((len(g) != 7 for _, g in groups)):
        raise ValueError('Experimental records do not match the registered condition layout.')
    y = np.stack([g[species].to_numpy(dtype=np.float64) for _, g in groups])
    t = np.stack([g['time'].to_numpy(dtype=np.float64) for _, g in groups])
    env = np.array([[g['temperature_C'].iloc[0] + 273.15, g['catalyst'].iloc[0]] for _, g in groups])
    if not np.isfinite(y).all() or (y < 0).any() or (not np.isfinite(t).all()):
        raise ValueError('Concentrations and residence times must be finite and nonnegative.')
    if not (np.diff(t, axis=1) > 0).all() or not np.all(t[:, 0] == 0):
        raise ValueError('Expected an inlet record followed by increasing residence times.')
    if any((g['temperature_C'].nunique() != 1 or g['catalyst'].nunique() != 1 for _, g in groups)):
        raise ValueError('Temperature and catalyst concentration must be constant within each condition.')
    if key in {'tbhp_wpo4', 'tbhp_cf3'}:
        stoich = np.array([[-1, 0], [-1, -1], [1, -1], [0, 1]], float)
        reactants = np.array([[0, 1], [1, 2]], np.int64)
    else:
        stoich = np.array([[-1], [-2 if key == 'tbpb_benzaldehyde' else -1], [1]], float)
        reactants = np.array([[0, 1]], np.int64)
    cref = []
    for a, b in reactants:
        positive_b = y[:, :, b][y[:, :, b] > 1e-08]
        cref.append([np.median(env[:, 1]), np.median(y[:, 0, a]), np.median(positive_b)])
    free = np.array([0, 1, 3, 4]) if key == 'tbhp_bzcl' else np.arange(5 * len(reactants))
    return Dataset(key, species, y, t, env, y[:, 0].copy(), stoich, reactants, np.array(cref), 1 / np.mean(1 / env[:, 0]), free, cfg['reference_start'], {'path': cfg['data_file'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'time_unit': 's', 'concentration_unit': 'mol/L'}, names)

def physical_to_scaled(data, physical):
    q = np.zeros(5 * len(data.reactants))
    for j, (a, b) in enumerate(data.reactants):
        i = j + 1
        if data.key == 'tbhp_bzcl':
            alpha, beta = (physical['order_H2O2'], physical['order_TBCL'])
        elif data.key == 'tbpb_benzaldehyde':
            alpha, beta = (physical['order_benzaldehyde'], physical['order_TBHP'])
        elif j == 0:
            alpha, beta = (physical['order_H2O2'], physical['order_TBA_main'])
        else:
            alpha, beta = (physical['order_TBA_side'], physical['order_TBHP_side'])
        m = physical.get(f'cat_order_{i}', 0.0)
        ea = physical[f'Ea{i}_over_R']
        log_ref = physical[f'lnA{i}'] - ea / data.tref + np.dot([m, alpha, beta], np.log(data.references[j]))
        q[5 * j:5 * j + 5] = [log_ref, ea * R / 100000, m, alpha, beta]
    return q

def scaled_to_physical(data, q):
    values = {}
    for j in range(len(data.reactants)):
        i = j + 1
        log_ref, escale, m, alpha, beta = q[5 * j:5 * j + 5]
        ea_over_r = escale * 100000 / R
        ln_a = log_ref + ea_over_r / data.tref - np.dot([m, alpha, beta], np.log(data.references[j]))
        values.update({f'lnA{i}': float(ln_a), f'Ea{i}_over_R': float(ea_over_r), f'Ea{i}_kJ_mol': float(escale * 100), f'log_rate_ref_{i}': float(log_ref)})
        if data.key != 'tbhp_bzcl':
            values[f'cat_order_{i}'] = float(m)
        if data.key == 'tbhp_bzcl':
            names = ['order_H2O2', 'order_TBCL']
        elif data.key == 'tbpb_benzaldehyde':
            names = ['order_benzaldehyde', 'order_TBHP']
        else:
            names = ['order_H2O2', 'order_TBA_main'] if j == 0 else ['order_TBA_side', 'order_TBHP_side']
        values.update(dict(zip(names, map(float, [alpha, beta]))))
    return values

def bounds(data, wide=False):
    low = np.tile([-30.0, 0.0001, 0.0, 0.05, 0.05], len(data.reactants))
    high = np.tile([10.0, 4.0 if wide else 2.5, 10.0 if wide else 6.0, 12.0 if wide else 8.0, 12.0 if wide else 8.0], len(data.reactants))
    return (low, high)

def _rates(y, log_k, q, reactants, refs):
    rates = np.zeros(len(reactants))
    for j in range(len(reactants)):
        a, b = reactants[j]
        if y[a] > 0 and y[b] > 0:
            value = log_k[j] + q[5 * j + 3] * math.log(y[a] / refs[j, 1]) + q[5 * j + 4] * math.log(y[b] / refs[j, 2])
            rates[j] = math.exp(min(80.0, max(-80.0, value)))
    return rates

def integrate_numpy(q, initial, times, conditions, stoich, reactants, refs, tref, max_step):
    ne, npnt = times.shape
    ns, nr = stoich.shape
    result = np.empty((ne, npnt, ns))
    for e in range(ne):
        y = initial[e].copy()
        result[e, 0] = y
        log_k = np.zeros(nr)
        for j in range(nr):
            log_k[j] = q[5 * j] - q[5 * j + 1] * 100000 / R * (1 / conditions[e, 0] - 1 / tref) + q[5 * j + 2] * math.log(conditions[e, 1] / refs[j, 0])
        for p in range(1, npnt):
            interval = times[e, p] - times[e, p - 1]
            steps = max(1, int(math.ceil(interval / max_step)))
            dt = interval / steps
            for _ in range(steps):
                a = _rates(y, log_k, q, reactants, refs)
                b = _rates(y + dt / 2 * (stoich @ a), log_k, q, reactants, refs)
                c = _rates(y + dt / 2 * (stoich @ b), log_k, q, reactants, refs)
                d = _rates(y + dt * (stoich @ c), log_k, q, reactants, refs)
                flux = dt * (a + 2 * b + 2 * c + d) / 6
                change = stoich @ flux
                factor = 1.0
                for n in range(ns):
                    if change[n] < 0:
                        factor = min(factor, max(0.0, y[n]) / -change[n])
                y = y + factor * change
            result[e, p] = y
    return result
_SOLVER = None

def integrate(q, initial, times, conditions, stoich, reactants, refs, tref, max_step):
    global _SOLVER
    if _SOLVER is None:
        from build_solver import build
        library = ctypes.CDLL(str(build()))
        _SOLVER = library.integrate_conservative
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        ip = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        _SOLVER.argtypes = [fp, fp, fp, fp, fp, ip, fp, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp]
        _SOLVER.restype = None
    ne, nt = times.shape
    ns, nr = stoich.shape
    if ns > 4 or nr > 2 or max_step <= 0:
        raise ValueError('Invalid solver dimensions or step.')
    result = np.empty((ne, nt, ns), dtype=np.float64)
    arrays = [np.ascontiguousarray(v, dtype=np.float64) for v in [q, initial, times, conditions, stoich]]
    _SOLVER(*arrays, np.ascontiguousarray(reactants, dtype=np.int64), np.ascontiguousarray(refs, dtype=np.float64), float(tref), float(max_step), ne, nt, ns, nr, result)
    return result

def predict(data, q, step=2.0):
    return integrate(np.asarray(q), data.initial, data.times, data.conditions, data.stoich, data.reactants, data.references, data.tref, float(step))

def fit(data, start, *, target=None, step=2.0, max_nfev=1000, wide=False, fixed=None, callback=None):
    target = data.observations if target is None else target
    lower, upper = bounds(data, wide)
    free = np.array([i for i in data.free if fixed is None or i != fixed[0]])
    base = np.asarray(start).copy()
    if fixed is not None:
        base[fixed[0]] = fixed[1]
    calls = 0

    def residual(x):
        nonlocal calls
        q = base.copy()
        q[free] = x
        values = (predict(data, q, step)[:, 1:] - target[:, 1:]).ravel()
        calls += 1
        if callback is not None and calls % 100 == 0:
            callback(calls, float(values @ values))
        return values
    solution = least_squares(residual, np.clip(base[free], lower[free] + 1e-08, upper[free] - 1e-08), bounds=(lower[free], upper[free]), jac='3-point', diff_step=0.0002, max_nfev=max_nfev, ftol=1e-09, xtol=1e-09, gtol=1e-08, x_scale='jac')
    q = base.copy()
    q[free] = solution.x
    return (q, {'success': bool(solution.success), 'status': int(solution.status), 'message': solution.message, 'nfev': int(solution.nfev), 'calls': calls, 'sse': float(solution.fun @ solution.fun), 'optimality': float(solution.optimality), 'jac_singular_values': np.linalg.svd(solution.jac, compute_uv=False).tolist()})

def metrics(data, prediction):
    results = {}
    for index, name in enumerate(data.species):
        for label, cut in [('all', 0), ('noninitial', 1)]:
            observed = data.observations[:, cut:, index].ravel()
            errors = prediction[:, cut:, index].ravel() - observed
            results[f'{name}_{label}'] = {'rmse': float(np.sqrt(np.mean(errors ** 2))), 'r2': float(1 - errors @ errors / np.sum((observed - observed.mean()) ** 2)), 'bias': float(errors.mean())}
    u, _, _ = np.linalg.svd(data.stoich, full_matrices=True)
    null = u[:, len(data.reactants):]
    error = (prediction - data.initial[:, None, :]) @ null
    results['physical_checks'] = {'min_concentration': float(prediction.min()), 'max_invariant_error': float(np.max(abs(error))), 'observed_max_invariant_error': float(np.max(abs((data.observations - data.initial[:, None, :]) @ null)))}
    return results

def export_predictions(data, prediction, path):
    rows = []
    for e in range(len(data.times)):
        for j, t in enumerate(data.times[e]):
            row = {'experiment_id': e, 'condition': data.sheets[e], 'time': t, 'temperature_C': data.conditions[e, 0] - 273.15, 'catalyst': data.conditions[e, 1]}
            for s, n in enumerate(data.species):
                row[n] = data.observations[e, j, s]
                row[n + '_pred'] = prediction[e, j, s]
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
