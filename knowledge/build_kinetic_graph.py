import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/kinetic_graph')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = load(REPO / 'kinetics/routes.json')
    metadata = load(ROOT / 'graph/route_metadata.json')
    models = {}
    nodes = []
    edges = []
    inputs = {}

    def node(identity, kind, route, basis, fields):
        nodes.append(dict(id=identity, kind=kind, route=route, basis=basis, fields=fields))

    for route, config in configs.items():
        source = REPO / 'kinetics' / config['data_file']
        rows = list(csv.DictReader(source.open()))
        inputs[str(source.relative_to(REPO))] = hashlib.sha256(source.read_bytes()).hexdigest()
        species = config['fitted_species']
        kinetic_columns = ['experiment_id', 'condition', 'time', 'temperature_C', 'catalyst'] + species
        grouped = defaultdict(list)
        for item in rows:
            grouped[item['condition']].append({k: item[k] if k == 'condition' else float(item[k]) for k in kinetic_columns})
        concentrations = species[:2]
        domain = {key: [min(x[key] for values in grouped.values() for x in values if x['time'] == 0), max(x[key] for values in grouped.values() for x in values if x['time'] == 0)] for key in ['temperature_C', 'catalyst'] + concentrations}
        domain['time_s'] = [0.0, max(x['time'] for values in grouped.values() for x in values)]
        final = load(REPO / 'kinetics/results' / route / 'final_summary.json')
        uncertainty = load(REPO / 'kinetics/results' / route / 'uncertainty.json')
        model = dict(route=route, metadata=metadata[route], parameters=final['parameters'], domain=domain,
                     domain_note='Marginal bounds only; unmeasured combinations are predictions, not automatically interpolation',
                     interpretation='Apparent empirical kinetics; not an elementary mechanism or peroxide decomposition kinetics',
                     units=dict(concentration='mol/L', time='s', temperature='degC (Arrhenius uses K)', Ea='kJ/mol'),
                     method=final['method'], uncertainty=uncertainty, conditions=dict(grouped))
        models[route] = model
        node('kinetics:' + route, 'KineticModel', route, 'fitted_apparent_parameter', {k: v for k, v in model.items() if k != 'conditions'})
        for condition, values in grouped.items():
            values.sort(key=lambda x: x['time'])
            assert len(values) == 7 and values[0]['time'] == 0
            experiment = f'experiment:{route}:{condition}'
            initial = {key: values[0][key] for key in species}
            node(experiment, 'Experiment', route, 'experiment', dict(condition=condition,
                 temperature_C=values[0]['temperature_C'], catalyst_mol_L=values[0]['catalyst'] if route != 'tbhp_bzcl' else None,
                 initial_mol_L=initial, concentration_unit='mol/L', time_unit='s', source=str(source.relative_to(REPO))))
            for index, row in enumerate(values):
                identity = f'observation:{route}:{condition}:{index}'
                node(identity, 'Observation', route, 'initial_state' if index == 0 else 'processed_experimental_state',
                     dict(time_s=row['time'], concentrations_mol_L={key: row[key] for key in species},
                          source=str(source.relative_to(REPO)), not_all_species_independently_measured=True))
                edges.append((experiment, 'HAS_OBSERVATION', identity))
        case = load(REPO / 'analysis/data/cases' / f'recommend_{route}.json')
        recommendation = next(x['recommendation'] for x in case['calculations'] if x['tool'] == 'recommend_conditions')
        node('ScreenedCondition:' + route, 'ScreenedCondition', route, 'model_prediction', recommendation)
    for line in (ROOT / 'graph/reactor_evidence.jsonl').read_text().splitlines():
        nodes.append(json.loads(line))
    destination = args.output_dir / 'graph.sqlite'
    if destination.exists():
        destination.unlink()
    with sqlite3.connect(destination) as connection:
        connection.execute('CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, route TEXT, evidence TEXT, payload TEXT)')
        connection.execute('CREATE TABLE edges (source TEXT, relation TEXT, target TEXT, PRIMARY KEY(source, relation, target))')
        connection.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)', [(n['id'], n['kind'], n['route'], n['basis'], json.dumps(n['fields'])) for n in nodes])
        connection.executemany('INSERT INTO edges VALUES (?,?,?)', edges)
    (args.output_dir / 'closed_loop_models.json').write_text(json.dumps(models, indent=2) + '\n')
    (args.output_dir / 'manifest.json').write_text(json.dumps(dict(inputs=inputs, nodes=len(nodes), edges=len(edges), conditions=60, states=420), indent=2) + '\n')
    print(json.dumps(dict(graph=str(destination), nodes=len(nodes), observations=420, conditions=60)))


if __name__ == '__main__':
    main()
