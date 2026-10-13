import runtime
import unittest
import numpy as np
import torch
from kinetic_core import CONFIGS, load_data, physical_to_scaled, scaled_to_physical, integrate, predict, metrics, _rates
from pinn_model import ExtentPINN

class PhysicsTests(unittest.TestCase):

    def test_analytic_bimolecular_solution(self):
        initial = np.array([[1.0, 1.0, 0.0]])
        times = np.array([[0.0, 0.5, 1.0, 5.0]])
        stoich = np.array([[-1.0], [-1.0], [1.0]])
        result = integrate(np.array([0.0, 0.1, 0.0, 1.0, 1.0]), initial, times, np.array([[300.0, 1.0]]), stoich, np.array([[0, 1]]), np.ones((1, 3)), 300.0, 0.005)
        np.testing.assert_allclose(result[0, :, 0], 1 / (1 + times[0]), atol=1e-10)

    def test_zero_reactant_has_zero_rate(self):
        result = _rates(np.array([0.0, 3.0, 2.0]), np.array([4.0]), np.array([4.0, 1.0, 1.0, 0.05, 0.05]), np.array([[0, 1]]), np.ones((1, 3)))
        self.assertEqual(result[0], 0.0)

    def test_exhaustion_preserves_stoichiometry(self):
        data = load_data('tbhp_wpo4')
        q = physical_to_scaled(data, data.initial_parameters)
        q[[0, 5]] += 8
        q[[3, 4, 8, 9]] = 0.1
        result = predict(data, q, 10)
        check = metrics(data, result)['physical_checks']
        self.assertGreaterEqual(check['min_concentration'], -1e-10)
        self.assertLess(check['max_invariant_error'], 1e-10)

    def test_parameter_roundtrip_and_network_hard_constraints(self):
        for key, cfg in CONFIGS.items():
            with self.subTest(route=key):
                data = load_data(key)
                physical = data.initial_parameters or dict(zip(cfg['param_names'], cfg['initial_params']))
                q = physical_to_scaled(data, physical)
                converted = scaled_to_physical(data, q)
                for name, value in physical.items():
                    self.assertAlmostEqual(value, converted[name], places=7)
                model = ExtentPINN(data, q, 123)
                tau = torch.linspace(0, 1, 31, dtype=torch.float64)[None, :, None].repeat(len(data.times), 1, 1)
                predicted = model.concentrations(tau).detach().numpy()
                self.assertGreaterEqual(predicted.min(), -1e-12)
                np.testing.assert_allclose(predicted[:, 0], data.initial, atol=1e-12)
                u, _, _ = np.linalg.svd(data.stoich, full_matrices=True)
                null = u[:, len(data.reactants):]
                self.assertLess(np.max(abs((predicted - data.initial[:, None]) @ null)), 1e-11)
if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
