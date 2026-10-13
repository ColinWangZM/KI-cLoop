import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from application_retrieval import retrieve_application_routes


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TBHP = 'TERT-BUTYL HYDROPEROXIDE'
TBPB = 'tert-butyl perbenzoate'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def names(result):
    require(result is not None, 'Application query was not handled')
    return {record['fields']['material'] for record in result['records']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path)
    args = parser.parse_args()
    index = json.loads((ROOT / 'data/material_application_index.json').read_text())
    source_rows = [json.loads(line) for line in (ROOT / 'data/application_route_sources.jsonl').read_text().splitlines()]
    sources = {row['source_id']: row for row in source_rows}
    require(len(index) == 2303, 'Unexpected candidate material count')
    require(len(sources) == len(source_rows), 'Duplicate source identifier')
    referenced = set()
    route_count = 0
    for material, entry in index.items():
        require(entry['material'] == material, 'Material identity mismatch')
        require(bool(entry['applications']), 'Missing application labels')
        require(bool(entry['application_label_source']), 'Missing application-label basis')
        for route in entry['synthesis_routes']:
            route_count += 1
            source_id = route['source_id']
            require(source_id in sources, 'Missing source: ' + source_id)
            referenced.add(source_id)
            source = sources[source_id]
            require(bool(source['source_context']), 'Missing source context: ' + source_id)
            require(bool(source['evidence_scope']), 'Missing evidence scope: ' + source_id)
            require(route['retrieval_status'] == source['retrieval_status'], 'Source status mismatch: ' + source_id)
            extracted_reactants = [name for context in source['source_context'] for name in context.get('reactant_names', [])]
            require(route['reactants'] == extracted_reactants, 'Reactants differ from source: ' + source_id)
            if route['route_source'] == 'organic_reaction_context':
                require(source['source_record']['node_name'] == route['product'], 'Product differs from source: ' + source_id)
            for record in source.get('repository_records', []):
                path = (REPOSITORY / record['path']).resolve()
                require(path.is_relative_to(REPOSITORY), 'Source path leaves repository')
                require(path.is_file(), 'Repository source missing: ' + record['path'])
                require(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], 'Repository source hash mismatch: ' + record['path'])
    require(route_count == 2310, 'Unexpected indexed route count')
    require(referenced == set(sources), 'Unused source records')
    require('TBCL' not in index['benzoyl chloride']['aliases'], 'TBCL incorrectly aliases benzoyl chloride')

    both_queries = [
        'Find synthesis routes for materials used as oxidizers or initiators.',
        'Which initiators have synthesis routes?',
        'What are the synthesis routes for TBHP and TBPB?',
        'How can tert-butyl hydroperoxide and tert-butyl peroxybenzoate be prepared?',
        '\u67e5\u8be2\u5f15\u53d1\u5242\u7684\u5408\u6210\u8def\u7ebf',
        '\u8bf7\u68c0\u7d22TBHP\u548cTBPB\u7684\u5408\u6210\u8def\u7ebf',
        '\u6709\u6c27\u5316\u5242\u548c\u5f15\u53d1\u5242\u5e94\u7528\u7684\u7269\u8d28\u6709\u54ea\u4e9b\u5408\u6210\u8def\u5f84',
        'initiators',
    ]
    checked = 0
    for question in both_queries:
        result = retrieve_application_routes(question)
        require({TBHP, TBPB} <= names(result), 'Target materials absent: ' + question)
        require(result['reference_answers_accessed'] is False, 'Reference answers used')
        returned_sources = {source['source_id'] for source in result['sources']}
        for record in result['records']:
            for route in record['fields']['synthesis_routes']:
                require(route['source_id'] in returned_sources, 'Returned route lacks source')
                require(route['retrieval_status'] != 'source_role_conflict', 'Conflicting source returned as a route')
        checked += 1
    for word in ['oxidants', 'oxidizers', 'oxidizing agents', 'oxidisers', '\u6c27\u5316\u5242']:
        result = retrieve_application_routes('Find synthesis routes for ' + word)
        require(TBHP in names(result) and TBPB not in names(result), 'Oxidant labels differ: ' + word)
        checked += 1
    for label, expected in [('TBHP', TBHP), ('TBPB', TBPB), ('tert-butyl peroxybenzoate', TBPB)]:
        require(names(retrieve_application_routes('How to synthesize ' + label + '?')) == {expected}, 'Chemical alias mismatch: ' + label)
        checked += 1
    for question in ['How to synthesize UNKNOWN_COMPOUND_X?', 'How to synthesize TPBP?', 'How to synthesize XTBHPX?']:
        require(names(retrieve_application_routes(question)) == set(), 'Unknown entity matched a material')
        checked += 1
    for question in ['For Path 1, what is the main rate multiplier from 40 to 60 degrees Celsius?', 'Describe Route 1 and Route 4.', 'Describe the reactor model and its coupled physical fields.', 'What is the activation energy for TBHP synthesis in Route 2?', 'Compare the yields for TBHP synthesis in Route 1 and Route 2.']:
        require(retrieve_application_routes(question) is None, 'Existing query intercepted: ' + question)
        checked += 1

    result = retrieve_application_routes('What are the synthesis routes for TBHP and TBPB?')
    require(names(result) == {TBHP, TBPB}, 'Direct query selected other materials')
    routes = {route['route_number']: route for record in result['records'] for route in record['fields']['synthesis_routes'] if route.get('route_source') == 'kinetics_experimental_route'}
    require(set(routes) == {1, 2, 3, 4}, 'Four experimental routes missing')
    require(set(routes[1]['reactants']) == {'hydrogen peroxide', 'tert-butyl chloride'}, 'Route 1 reactants differ')
    require(routes[2]['catalysts'] == ['phosphotungstic acid (PTA)'], 'Route 2 catalyst differs')
    require(routes[3]['catalysts'] == ['trifluoroacetic acid (TFA)'], 'Route 3 catalyst differs')
    require(set(routes[4]['reactants']) == {'benzaldehyde', 'tert-butyl hydroperoxide'}, 'Route 4 reactants differ')
    require(routes[4]['catalysts'] == ['tetrabutylammonium iodide (TBAI)'] and routes[4]['solvents'] == ['acetonitrile'], 'Route 4 catalyst or solvent differs')

    with tempfile.TemporaryDirectory() as temporary:
        command = [sys.executable, '-S', str(ROOT / 'query.py'), 'What are the synthesis routes for TBHP and TBPB?', '--graph', str(Path(temporary) / 'absent.sqlite')]
        process = subprocess.run(command, cwd=temporary, capture_output=True, text=True, check=True)
        require(names(json.loads(process.stdout)) == {TBHP, TBPB}, 'Public entry requires external files or site packages')
    if args.graph:
        command = [sys.executable, str(ROOT / 'query.py'), 'Describe Route 1, Route 2, Route 3 and Route 4.', '--graph', str(args.graph.resolve())]
        process = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, check=True)
        expected = {'kinetics:tbhp_bzcl', 'kinetics:tbhp_wpo4', 'kinetics:tbhp_cf3', 'kinetics:tbpb_benzaldehyde'}
        require({row['id'] for row in json.loads(process.stdout)['records']} == expected, 'Existing route queries differ')
        command[2] = 'What is the activation energy for TBHP synthesis in Route 2?'
        process = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, check=True)
        require({row['id'] for row in json.loads(process.stdout)['records']} == {'kinetics:tbhp_wpo4'}, 'Mixed synthesis and kinetic query selected the wrong evidence')
    print(json.dumps({'materials': len(index), 'indexed_routes': route_count, 'source_records': len(sources), 'query_checks': checked, 'public_entry_without_graph_or_site_packages': 'passed', 'kinetic_graph_entry': 'passed' if args.graph else 'not requested', 'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
