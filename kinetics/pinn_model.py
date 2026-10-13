from __future__ import annotations
import runtime
import copy
import json
import time
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from torch import nn
from kinetic_core import bounds, predict, scaled_to_physical, metrics

class ExtentPINN(nn.Module):

    def __init__(self, data, initial_q, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.double = len(data.reactants) == 2
        self.tbpb = data.key == 'tbpb_benzaldehyde'
        ne = len(data.times)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for ni, no in [(2, 24), (24, 24), (24, len(data.reactants))]:
            w = torch.empty(ne, ni, no, dtype=torch.float64)
            nn.init.xavier_normal_(w)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(ne, 1, no, dtype=torch.float64)))
        for name, value in [('initial', data.initial), ('stoich', data.stoich), ('conditions', data.conditions), ('refs', data.references), ('full_q', initial_q), ('tmax', data.times[:, -1])]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        self.free = list(map(int, data.free))
        low, high = bounds(data)
        self.register_buffer('low', torch.tensor(low[data.free], dtype=torch.float64))
        self.register_buffer('high', torch.tensor(high[data.free], dtype=torch.float64))
        fraction = np.clip((initial_q[data.free] - low[data.free]) / (high[data.free] - low[data.free]), 1e-08, 1 - 1e-08)
        self.raw_q = nn.Parameter(torch.tensor(np.log(fraction / (1 - fraction)), dtype=torch.float64))
        self.tref = data.tref
        self.reactants = data.reactants.tolist()

    @property
    def q(self):
        q = self.full_q.clone()
        q[self.free] = self.low + (self.high - self.low) * torch.sigmoid(self.raw_q)
        return q

    def extents(self, tau):
        x = torch.cat([tau, torch.log1p(100 * tau) / np.log(101)], dim=-1)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, w) + b
            if i < 2:
                x = torch.tanh(x)
        clocks = tau.expand_as(x)
        if self.double:
            delta = self.q[9]
            side_clock = (tau[:, :, 0] + 1e-12) ** delta - 1e-12 ** delta
            clocks = torch.stack([tau[:, :, 0], side_clock], dim=-1)
        fractions = -torch.expm1(-clocks * torch.exp(torch.clamp(x, -20.0, 20.0)))
        a = self.initial[:, 0:1]
        b = self.initial[:, 1:2]
        cap = torch.minimum(a, b / (2 if self.tbpb else 1))
        first = cap * fractions[:, :, 0]
        if not self.double:
            return first[:, :, None]
        remaining = torch.minimum(b - first, self.initial[:, 2:3] + first)
        second = remaining * fractions[:, :, 1]
        return torch.stack([first, second], dim=-1)

    def concentrations(self, tau):
        return self.initial[:, None, :] + self.extents(tau) @ self.stoich.T

    def rates(self, c):
        q = self.q
        results = []
        for j, (a, b) in enumerate(self.reactants):
            logk = q[5 * j] - q[5 * j + 1] * 100000 / 8.314 * (1 / self.conditions[:, 0] - 1 / self.tref) + q[5 * j + 2] * torch.log(self.conditions[:, 1] / self.refs[j, 0])
            ca, cb = (c[:, :, a], c[:, :, b])
            logr = logk[:, None] + q[5 * j + 3] * torch.log(torch.clamp(ca / self.refs[j, 1], min=1e-30)) + q[5 * j + 4] * torch.log(torch.clamp(cb / self.refs[j, 2], min=1e-30))
            rate = torch.where((ca > 0) & (cb > 0), torch.exp(torch.clamp(logr, -80, 80)), 0.0)
            results.append(rate)
        return torch.stack(results, dim=-1)

def train_pinn(data, initial_q, out, *, epochs=10000, warmup=1500, lbfgs=600, seed=20260917, log_every=100, progress=None, step=2.0):
    torch.set_num_threads(1)
    model = ExtentPINN(data, np.asarray(initial_q), seed)
    network_params = list(model.weights) + list(model.biases)
    tau_data = torch.tensor(data.times / data.times[:, -1:], dtype=torch.float64)[:, :, None]
    observed = torch.tensor(data.observations, dtype=torch.float64)
    scale = max(float(data.observations.std()), 0.1)
    grid = np.unique(np.r_[0.0, np.geomspace(1e-06, 1, 48), np.linspace(0, 1, 48), (data.times / data.times[:, -1:]).ravel()])
    dense = replace(data, times=data.times[:, -1:] * grid[None, :])
    ode_teacher = torch.tensor(predict(dense, initial_q, step), dtype=torch.float64)
    tau_dense = torch.tensor(np.broadcast_to(grid[None, :, None], (len(data.times), len(grid), 1)).copy(), dtype=torch.float64)
    quadrature = np.zeros_like(grid)
    quadrature[:-1] += np.diff(grid) / 2
    quadrature[1:] += np.diff(grid) / 2
    quadrature = torch.tensor(quadrature, dtype=torch.float64)[None, :, None]
    opt = torch.optim.Adam(network_params, lr=0.01)
    history = []
    started = time.monotonic()

    def record(stage, iteration, total, data_loss, physics_loss, loss):
        row = {'stage': stage, 'iteration': iteration, 'total': total, 'data_loss': float(data_loss), 'physics_loss': float(physics_loss), 'loss': float(loss), 'elapsed_s': time.monotonic() - started}
        row.update(scaled_to_physical(data, model.q.detach().numpy()))
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'pinn_history.csv', index=False)
        if progress:
            progress(stage, iteration, total, f'data={data_loss:.4g} physics={physics_loss:.4g}')
    for epoch in range(1, warmup + 1):
        opt.zero_grad()
        loss = torch.mean(((model.concentrations(tau_dense) - ode_teacher) / scale) ** 2)
        loss.backward()
        opt.step()
        if epoch % log_every == 0 or epoch == warmup:
            record('pinn_warmup', epoch, warmup, loss.item(), 0.0, loss.item())
    opt = torch.optim.Adam([{'params': network_params, 'lr': 0.003}, {'params': [model.raw_q], 'lr': 0.00015}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=400, factor=0.5, min_lr=1e-06)
    best = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    physics_weight = 1.0
    teacher_weight = 1.0

    def losses():
        collocation = tau_dense.detach().clone().requires_grad_(True)
        xi = model.extents(collocation)
        concentration = model.initial[:, None, :] + xi @ model.stoich.T
        derivatives = torch.stack([torch.autograd.grad(xi[:, :, j].sum(), collocation, create_graph=True, retain_graph=True)[0][:, :, 0] for j in range(xi.shape[-1])], dim=-1)
        residual = (derivatives - model.tmax[:, None, None] * model.rates(concentration)) / scale
        physical = torch.mean(torch.sum(residual ** 2 * quadrature, dim=1))
        data_loss = torch.mean(((model.concentrations(tau_data)[:, 1:] - observed[:, 1:]) / scale) ** 2)
        teacher_loss = torch.mean(torch.sum(((concentration - ode_teacher) / scale) ** 2 * quadrature, dim=1))
        return (data_loss, physical, data_loss + physics_weight * physical + teacher_weight * teacher_loss)
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        dl, pl, loss = losses()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite PINN loss; saved outputs remain intact.')
        score = loss.item()
        if score < best * (1 - 1e-07):
            best = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        scheduler.step(score)
        if epoch % log_every == 0 or epoch == epochs:
            record('pinn_joint', epoch, epochs, dl.item(), pl.item(), score)
        if epoch >= min(6000, epochs) and epoch - best_epoch >= 2000:
            break
    model.load_state_dict(best_state)
    closure_calls = 0
    quasi = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=lbfgs, tolerance_grad=1e-09, tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        nonlocal closure_calls, best, best_state
        quasi.zero_grad()
        dl, pl, loss = losses()
        loss.backward()
        closure_calls += 1
        if loss.item() < best:
            best = loss.item()
            best_state = copy.deepcopy(model.state_dict())
        if closure_calls % 50 == 0:
            record('pinn_lbfgs', closure_calls, lbfgs, dl.item(), pl.item(), loss.item())
        return loss
    if lbfgs > 0:
        quasi.step(closure)
    model.load_state_dict(best_state)
    dl, pl, loss = losses()
    learned = model.q.detach().numpy()
    prediction = model.concentrations(tau_data).detach().numpy()
    ode_prediction = predict(data, learned, step)
    summary = {'method': 'joint_concentration_network_PINN', 'seed': seed, 'warmup_source': 'same-data mechanistic ODE fit, explicitly not independent', 'adam_epochs_executed': epoch, 'adam_epochs_requested': epochs, 'lbfgs_closure_calls': closure_calls, 'best_adam_epoch': best_epoch, 'physics_weight': physics_weight, 'ODE_teacher_weight': teacher_weight, 'physical_residual_time_quadrature': True, 'data_loss': dl.item(), 'physics_loss': pl.item(), 'q': learned.tolist(), 'parameters': scaled_to_physical(data, learned), 'network_metrics': metrics(data, prediction), 'forward_ode_metrics': metrics(data, ode_prediction), 'max_network_ode_gap': float(np.max(abs(prediction - ode_prediction))), 'elapsed_s': time.monotonic() - started}
    torch.save({'state_dict': model.state_dict(), 'q': learned, 'source': data.source}, out / 'pinn_network.pt')
    (out / 'pinn_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    record('pinn_complete', epoch, epochs, dl.item(), pl.item(), loss.item())
    return (learned, summary, prediction)
