import json
import math
import re
import sqlite3
from pathlib import Path


ROUTES = ['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3', 'tbpb_benzaldehyde']


def orders(fields):
    p = fields['parameters']
    if fields['route'] == 'tbhp_bzcl':
        return {'H2O2': p['order_H2O2'], 'TBCL': p['order_TBCL']}
    if fields['route'] == 'tbpb_benzaldehyde':
        return {'Benzaldehyde': p['order_benzaldehyde'], 'TBHP': p['order_TBHP'],
                'catalyst': p['cat_order_1']}
    return {'H2O2': p['order_H2O2'], 'TBA': p['order_TBA_main'],
            'catalyst': p['cat_order_1']}


class Evidence:
    def __init__(self, graph_path):
        self.graph_path = Path(graph_path).expanduser().resolve()
        self.db = sqlite3.connect(self.graph_path.as_uri() + '?mode=ro', uri=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def node(self, identity):
        row = self.db.execute('SELECT kind,route,evidence,payload FROM nodes WHERE id=?',
                              (identity,)).fetchone()
        if row is None:
            raise KeyError(identity)
        return dict(id=identity, kind=row[0], route=row[1], basis=row[2], fields=json.loads(row[3]))

    def experiment(self, route, condition):
        identity = f'experiment:{route}:{condition}'
        record = self.node(identity)
        observations = [self.node(row[0]) for row in self.db.execute(
            "SELECT target FROM edges WHERE source=? AND relation='HAS_OBSERVATION'", (identity,))]
        observations.sort(key=lambda item: item['fields']['time_s'])
        record['observations'] = observations
        return record

    def endpoint(self, record):
        fields = record['fields']
        initial = fields['initial_mol_L']
        end = record['observations'][-1]['fields']
        values = end['concentrations_mol_L']
        route = record['route']
        if route == 'tbpb_benzaldehyde':
            limit = min(initial['Benzaldehyde'], initial['TBHP'] / 2)
            product = 'TBPB'
            denominator = initial['Benzaldehyde'] - values['Benzaldehyde']
        else:
            limit = min(initial['H2O2'], initial['TBCL' if route == 'tbhp_bzcl' else 'TBA'])
            product = 'TBHP'
            denominator = (values['TBHP'] + values['DTBP'] if 'DTBP' in values
                           else initial['H2O2'] - values['H2O2'])
        return dict(condition=fields['condition'], temperature_C=fields['temperature_C'],
                    time_s=end['time_s'], product=product, concentrations_mol_L=values,
                    yield_pct=100 * values[product] / limit,
                    selectivity_pct=100 * values[product] / denominator if denominator > 1e-12 else None,
                    limiting_initial_mol_L=limit,
                    formula='yield=product/min(initial substrate, stoichiometric oxidant capacity); '
                            'selectivity=TBHP/(TBHP+DTBP) for Paths 2/3; '
                            'TBPB/benzaldehyde consumed for Path 4')

    def retrieve(self, question):
        if re.search(r'COMSOL|tube|heat.transfer|coolant|parallel|axisymmetric|volumetric heat|'
                     r'wall area|reactor (?:model|geometry|simulation)', question, re.I):
            overview = self.node('engineering:comsol')
            overview['fields'] = {key: value for key, value in overview['fields'].items()
                                  if key in {'geometry', 'type', 'physics', 'coupling', 'qualitative_role'}}
            return dict(records=[overview], calculations=[], source_graph=str(self.graph_path))

        indices = sorted({int(value) - 1 for value in re.findall(r'(?:Path|Route)\s+([1-4])', question, re.I)})
        records, calculations = [], []
        for index in indices:
            route = ROUTES[index]
            model = self.node('kinetics:' + route)
            fields = model['fields']
            parameters = fields['parameters']
            main_orders = orders(fields)
            model['fields'] = {key: fields[key] for key in
                               ['route', 'metadata', 'parameters', 'domain', 'interpretation', 'units']}
            records.append(model)
            if re.search('screen|recommend|optimum', question, re.I):
                records.append(self.node('ScreenedCondition:' + route))
            for (identity,) in self.db.execute("SELECT id FROM nodes WHERE kind='Experiment' AND route=?", (route,)):
                condition = identity.split(':', 2)[2]
                if re.search(r'(?<![\d.\-])' + re.escape(condition) + r'(?![\d-]|\.\d)', question):
                    record = self.experiment(route, condition)
                    records.append(record)
                    calculations.append(dict(tool='recorded_endpoint_metrics', source=identity,
                                             result=self.endpoint(record)))
            temperature_pattern = r'from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)\s*(?:°\s*C|degrees?\s+Celsius)'
            for low, high in re.findall(temperature_pattern, question, re.I):
                low, high = float(low), float(high)
                inverse_difference = 1 / (low + 273.15) - 1 / (high + 273.15)
                factors = {f'r{channel}_high_over_low': math.exp(parameters[f'Ea{channel}_over_R'] * inverse_difference)
                           for channel in [1, 2] if f'Ea{channel}_over_R' in parameters}
                if 'r2_high_over_low' in factors:
                    factors['side_over_main_multiplier'] = factors['r2_high_over_low'] / factors['r1_high_over_low']
                calculations.append(dict(tool='Arrhenius', source=model['id'], temperatures_C=[low, high],
                                         formula='exp((Ea/R)*(1/T_low-1/T_high)); temperatures converted to K',
                                         result=factors, scope='fixed instantaneous composition; not endpoint yield'))
            side_orders = ({'catalyst': parameters['cat_order_2'], 'TBA': parameters['order_TBA_side'],
                            'TBHP': parameters['order_TBHP_side'], 'H2O2': 0.}
                           if 'Ea2_over_R' in parameters else {})
            sensitivity_pattern = r'(H2O2|TBCL|TBA|TBHP|Benzaldehyde|catalyst|all concentrations) × (\d+(?:\.\d+)?)'
            for species, multiplier in re.findall(sensitivity_pattern, question):
                multiplier = float(multiplier)
                main_order = sum(main_orders.values()) if species == 'all concentrations' else main_orders.get(species, 0.)
                side_order = sum(side_orders.values()) if species == 'all concentrations' else side_orders.get(species, 0.)
                result = dict(main_order=main_order, main_rate_multiplier=multiplier ** main_order)
                if side_orders:
                    result.update(side_order=side_order, side_rate_multiplier=multiplier ** side_order,
                                  side_over_main_exponent=side_order - main_order,
                                  side_over_main_multiplier=multiplier ** (side_order - main_order))
                calculations.append(dict(tool='power_law_sensitivity', source=model['id'], species=species,
                                         concentration_multiplier=multiplier, formula='r_new/r_old=multiplier**order',
                                         result=result, scope='temperature and other instantaneous concentrations fixed; not endpoint yield'))
            if side_orders:
                calculations.append(dict(tool='side_main_ratio_exponents', source=model['id'],
                                         result={species: side_orders.get(species, 0) - main_orders.get(species, 0)
                                                 for species in sorted(set(main_orders) | set(side_orders))},
                                         delta_Ea_side_minus_main_kJ_mol=parameters['Ea2_kJ_mol'] - parameters['Ea1_kJ_mol']))
        return dict(records=records, calculations=calculations, source_graph=str(self.graph_path))
