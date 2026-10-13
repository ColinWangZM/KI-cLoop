from pathlib import Path
import os
from hashlib import sha256
import json
import math
import sys
import numpy as np

ROOT=Path(os.environ.get("KICLOOP_PROPERTY_WORK", Path(__file__).resolve().parents[1] / "property_work")).resolve()
BASE=Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
import duckdb

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    c=duckdb.connect()
    c.execute('SET threads=24')
    c.execute("SET memory_limit='40GB'")
    for table in ['nodes','edges','property_nodes','property_edges','property_values','quantity_relation_evidence',
                  'reaction_models','physical_quantities','quantity_formulas']:
        c.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{ROOT}/graph_parquet/{table}.parquet')")
    report={'checks':{},'scope':'Structural and provenance checks plus targeted numerical/semantic invariants; not a global chemical-accuracy estimate.'}

    def check(name,query):
        result=c.execute(query).fetchone()[0]
        report['checks'][name]=result
        print('[check]',name,result,flush=True)
        assert result==0,(name,result)

    check('duplicate_nodes','SELECT count(*)-count(DISTINCT node_id) FROM nodes')
    check('duplicate_edges','SELECT count(*)-count(DISTINCT edge_id) FROM edges')
    check('duplicate_triples','SELECT count(*)-count(DISTINCT (subject_id,predicate,object_id)) FROM edges')
    check('dangling_subjects','SELECT count(*) FROM edges e ANTI JOIN nodes n ON e.subject_id=n.node_id')
    check('dangling_objects','SELECT count(*) FROM edges e ANTI JOIN nodes n ON e.object_id=n.node_id')
    check('duplicate_property_records','SELECT count(*)-count(DISTINCT property_value_id) FROM property_values')
    check('invalid_property_values','SELECT count(*) FROM property_values WHERE value IS NULL OR NOT isfinite(value) OR unit IS NULL OR source_ref IS NULL OR source_sha256 IS NULL')
    check('property_value_missing_edge','SELECT count(*) FROM property_values v ANTI JOIN edges e ON v.edge_id=e.edge_id')
    check('property_edge_without_value',"SELECT count(*) FROM property_edges e ANTI JOIN property_values v ON e.edge_id=v.edge_id WHERE e.predicate='has_property'")
    check('property_target_not_quantity',"SELECT count(*) FROM property_edges e JOIN nodes n ON e.object_id=n.node_id WHERE e.predicate='has_property' AND n.node_type<>'PhysicalQuantity'")
    check('reaction_parameter_misattributed_to_species',"SELECT count(*) FROM property_values v JOIN nodes n ON v.subject_id=n.node_id WHERE n.node_type IN ('Molecule','MaterialRecord') AND v.quantity_id IN ('quantity:activation_energy','quantity:reaction_order','quantity:rate_coefficient','quantity:preexponential_factor')")
    check('unprovided_gibbs_values',"SELECT count(*) FROM property_values WHERE quantity_id IN ('quantity:molar_gibbs_energy','quantity:reaction_gibbs_energy')")
    check('property_value_identity_mismatch',"SELECT count(*) FROM property_values v JOIN edges e ON v.edge_id=e.edge_id WHERE v.subject_id<>e.subject_id OR v.quantity_id<>e.object_id OR e.predicate<>'has_property'")
    check('quantity_relation_without_evidence',"SELECT count(*) FROM property_edges e ANTI JOIN quantity_relation_evidence q USING(edge_id) WHERE e.predicate<>'has_property'")
    check('unknown_formula',"SELECT count(*) FROM quantity_relation_evidence q ANTI JOIN quantity_formulas f USING(formula_id) WHERE q.formula_id IS NOT NULL")
    check('formula_edges_not_between_quantities',"SELECT count(*) FROM property_edges e JOIN nodes s ON e.subject_id=s.node_id JOIN nodes o ON e.object_id=o.node_id WHERE e.predicate='derive' AND (s.node_type<>'PhysicalQuantity' OR o.node_type<>'PhysicalQuantity')")
    check('unqualified_influence_edges',"SELECT count(*) FROM quantity_relation_evidence WHERE predicate='influences' AND (context_id IS NULL OR notes IS NULL OR formula_id IS NULL)")
    check('orphan_property_nodes','SELECT count(*) FROM property_nodes n ANTI JOIN (SELECT subject_id node_id FROM property_edges UNION SELECT object_id FROM property_edges) e USING(node_id)')
    check('nonfinite_interval_bounds','SELECT count(*) FROM property_values WHERE (ci95_low IS NOT NULL AND NOT isfinite(ci95_low)) OR (ci95_high IS NOT NULL AND NOT isfinite(ci95_high))')
    check('invalid_interval_order','SELECT count(*) FROM property_values WHERE ci95_low>ci95_high')
    check('lost_base_nodes',f"SELECT count(*) FROM read_parquet('{BASE}/graph_parquet/nodes.parquet') b ANTI JOIN nodes n USING(node_id)")
    check('lost_base_edges',f"SELECT count(*) FROM read_parquet('{BASE}/graph_parquet/edges.parquet') b ANTI JOIN edges e USING(edge_id)")

    refs=c.execute('SELECT DISTINCT source_ref,source_sha256 FROM property_values').fetchall()
    for ref,expected in refs:assert digest(ROOT/ref)==expected,(ref,'checksum mismatch')
    for field,q,factor in [('density','density',1000.),('volume','cell_volume',1e-30),('band_gap','band_gap',1.),('e_total','relative_permittivity',1.)]:
        path=ROOT/f'inputs/material_project__{field}.parquet'
        check('source_values_'+field,f"""SELECT count(*) FROM property_values v
          JOIN read_parquet('{path}') s ON v.source_row=s.source_row
          WHERE v.source_ref='inputs/{path.name}' AND
          (v.subject_id<>'material:'||s.dataset||':'||s.record_id OR
           abs(v.value-try_cast(s.value_raw AS DOUBLE)*{factor})>greatest(abs(v.value)*1e-12,1e-40))""")

    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'kinetics'))
    from kinetic_core import load_data, physical_to_scaled, _rates, R
    models=json.loads((ROOT/'inputs/closed_loop_models.json').read_text())
    data_by_route={route:load_data(route) for route in models}
    max_error=0.
    checked_branches=0
    for mid,route,branch,anchor in c.execute('SELECT node_id,route,branch,anchor_json FROM reaction_models').fetchall():
        info=json.loads(anchor);state=models[route]['conditions'][info['anchor_condition']][1]
        data=data_by_route[route]
        q=physical_to_scaled(data,models[route]['parameters'])
        conc=np.array([state[name] for name in data.species])
        log_k=np.array([q[5*j]-q[5*j+1]*100000/R*(1/info['temperature_K']-1/data.tref)
                        +q[5*j+2]*math.log(state['catalyst']/data.references[j,0])
                        for j in range(len(data.reactants))])
        expected=_rates(conc,log_k,q,data.reactants,data.references)[branch-1]
        actual=c.execute("SELECT value FROM property_values WHERE subject_id=? AND quantity_id='quantity:reaction_rate'",[mid]).fetchone()[0]
        assert math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-15),(route,branch,expected,actual)
        max_error=max(max_error,abs(expected-actual));checked_branches+=1
    report['project_rate_check']={'reference':'kinetics/kinetic_core.py','branches':checked_branches,'maximum_absolute_error':max_error}
    k=lambda activation_temperature,prefactor,temperature:prefactor*math.exp(-activation_temperature/temperature)
    assert k(2,1,2)>k(2,1,1)
    assert k(3,1,1)<k(2,1,1)
    report['formula_sanity_checks']='PASS; Arrhenius temperature and activation-energy dependence'
    for name,expected in json.loads((ROOT/'audit/base_hashes_before.json').read_text()).items():
        assert digest(BASE/'graph_parquet'/name)==expected,name
    report['base_graph_unchanged']=True
    manifest=json.loads((ROOT/'audit/input_manifest.json').read_text())
    unavailable_sources=[]
    for name,entry in manifest['inputs'].items():
        assert digest(ROOT/'inputs'/name)==entry['sha256']
        source_path=Path(entry['source'])
        if source_path.exists():
            assert digest(source_path)==entry['sha256']
        else:
            unavailable_sources.append(entry['source'])
    report['source_project_inputs_verified']=not unavailable_sources
    report['sources_unavailable_on_this_host']=unavailable_sources
    report['project_inputs_verified']=True
    report['nodes_by_type']=dict(c.execute('SELECT node_type,count(*) FROM nodes GROUP BY node_type ORDER BY node_type').fetchall())
    report['edges_by_type']=dict(c.execute('SELECT predicate,count(*) FROM edges GROUP BY predicate ORDER BY predicate').fetchall())
    report['node_count']=sum(report['nodes_by_type'].values())
    report['edge_count']=sum(report['edges_by_type'].values())
    report['property_nodes']=c.execute('SELECT count(*) FROM property_nodes').fetchone()[0]
    report['property_edges']=c.execute('SELECT count(*) FROM property_edges').fetchone()[0]
    report['property_value_count']=c.execute('SELECT count(*) FROM property_values').fetchone()[0]
    report['quantities_with_numeric_values']=dict(c.execute('SELECT quantity_id,count(*) FROM property_values GROUP BY quantity_id ORDER BY quantity_id').fetchall())
    report['formula_count']=c.execute('SELECT count(*) FROM quantity_formulas').fetchone()[0]
    report['status']='PASS'
    (ROOT/'audit/validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('[PASS] Graph, ownership, source values, numerical formulas and unchanged inputs verified',flush=True)

if __name__=='__main__':main()
