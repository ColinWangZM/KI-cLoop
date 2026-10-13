from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import os
import json
import sys
import time
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
import duckdb

def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    start=time.time()
    c=duckdb.connect()
    c.execute('SET threads=24')
    c.execute("SET memory_limit='40GB'")
    for name in ['nodes','edges','molecules','materials','reactions','reaction_sources',
                 'reaction_edge_evidence','source_row_disposition','classes']:
        c.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{ROOT}/graph_parquet/{name}.parquet')")
    c.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{ROOT}/raw_parquet/uspto/*.parquet')")
    report={'checks':{},'count_definition':'Unique typed nodes and unique directed subject–predicate–object relations. '
            'Material records are source-scoped; reaction records group patent plus exact extracted procedure. '
            'These are not counts of globally unique substances or unique reaction transformations.'}

    def check(name,query):
        value=c.execute(query).fetchone()[0]
        report['checks'][name]=value
        print(f'[check] {name}: {value}',flush=True)
        assert value==0,(name,value)

    report['node_count']=c.execute('SELECT count(*) FROM nodes').fetchone()[0]
    report['edge_count']=c.execute('SELECT count(*) FROM edges').fetchone()[0]
    report['nodes_by_type']=dict(c.execute('SELECT node_type,count(*) FROM nodes GROUP BY node_type ORDER BY node_type').fetchall())
    report['edges_by_type']=dict(c.execute('SELECT predicate,count(*) FROM edges GROUP BY predicate ORDER BY predicate').fetchall())
    report['primary_domain_node_count']=sum(report['nodes_by_type'][key] for key in ['Molecule','MaterialRecord','ReactionRecord'])
    report['domain_edge_count_excluding_provenance']=sum(value for key,value in report['edges_by_type'].items() if key not in ['recorded_in','reported_in'])
    assert report['primary_domain_node_count']>=2_000_000
    assert report['domain_edge_count_excluding_provenance']>=4_000_000
    print('[counts]',report['node_count'],report['edge_count'],flush=True)
    check('duplicate_node_ids','SELECT count(*)-count(DISTINCT node_id) FROM nodes')
    check('duplicate_edge_ids','SELECT count(*)-count(DISTINCT edge_id) FROM edges')
    check('duplicate_edge_triples','SELECT count(*)-count(DISTINCT (subject_id,predicate,object_id)) FROM edges')
    check('null_node_ids',"SELECT count(*) FROM nodes WHERE node_id IS NULL OR length(node_id)=0")
    check('null_edge_endpoints','SELECT count(*) FROM edges WHERE subject_id IS NULL OR object_id IS NULL OR predicate IS NULL')
    check('dangling_subjects','SELECT count(*) FROM edges e ANTI JOIN nodes n ON e.subject_id=n.node_id')
    check('dangling_objects','SELECT count(*) FROM edges e ANTI JOIN nodes n ON e.object_id=n.node_id')
    check('duplicate_source_rows','SELECT count(*)-count(DISTINCT (source_file,source_row)) FROM source_row_disposition')
    check('missing_source_row_dispositions','SELECT count(*) FROM raw r ANTI JOIN source_row_disposition d USING(source_file,source_row)')
    check('unresolved_reaction_source_rows','SELECT count(*) FROM reaction_sources s ANTI JOIN raw r USING(source_file,source_row)')
    check('role_edges_without_row_support',"SELECT count(*) FROM edges e ANTI JOIN (SELECT DISTINCT edge_id FROM reaction_edge_evidence) s USING(edge_id) WHERE e.predicate IN ('has_reactant','has_product','has_solvent')")
    check('role_evidence_without_edges','SELECT count(*) FROM reaction_edge_evidence s ANTI JOIN edges e USING(edge_id)')
    check('reaction_without_product','SELECT count(*) FROM reactions WHERE product_count=0')
    check('reaction_without_reactant','SELECT count(*) FROM reactions WHERE reactant_count=0')
    check('duplicate_material_records','SELECT count(*)-count(DISTINCT (dataset,record_id)) FROM materials')
    check('duplicate_molecule_inchi','SELECT count(*)-count(DISTINCT canonical_inchi) FROM molecules')
    check('class_edges_target_nonclass',"SELECT count(*) FROM edges e JOIN nodes n ON e.object_id=n.node_id WHERE predicate='belongs_to' AND n.node_type<>'StructuralClass'")
    check('role_edges_target_nonmolecule',"SELECT count(*) FROM edges e JOIN nodes n ON e.object_id=n.node_id WHERE predicate IN ('has_reactant','has_product','has_solvent') AND n.node_type<>'Molecule'")
    c.execute(f"CREATE VIEW structures AS SELECT dataset,record_id FROM read_parquet('{ROOT}/raw_parquet/structures/*.parquet')")
    c.execute(f"CREATE VIEW properties AS SELECT dataset,record_id,property_name,value_raw FROM read_parquet('{ROOT}/raw_parquet/materials/*.parquet')")
    check('material_records_without_cif','SELECT count(*) FROM materials ANTI JOIN structures USING(dataset,record_id)')
    check('cif_without_material_records','SELECT count(*) FROM structures ANTI JOIN materials USING(dataset,record_id)')
    check('property_rows_without_material_records','SELECT count(*) FROM properties ANTI JOIN materials USING(dataset,record_id)')
    report['source_rows']=c.execute('SELECT count(*) FROM raw').fetchone()[0]
    report['source_row_dispositions']=dict(c.execute('SELECT disposition,count(*) FROM source_row_disposition GROUP BY disposition ORDER BY disposition').fetchall())
    report['materials_by_dataset']=dict(c.execute('SELECT dataset,count(*) FROM materials GROUP BY dataset ORDER BY dataset').fetchall())
    report['material_property_rows']=c.execute('SELECT count(*) FROM properties').fetchone()[0]
    report['cif_records']=c.execute('SELECT count(*) FROM structures').fetchone()[0]
    report['organic_peroxide_molecule_count']=c.execute("SELECT count(*) FROM molecules WHERE list_contains(class_ids,'class:molecule:organic_peroxide')").fetchone()[0]
    report['reactions_with_organic_peroxide_role']=c.execute("SELECT count(DISTINCT e.subject_id) FROM edges e JOIN molecules m ON e.object_id=m.node_id WHERE e.predicate IN ('has_reactant','has_product') AND list_contains(m.class_ids,'class:molecule:organic_peroxide')").fetchone()[0]
    report['distinct_recorded_reactant_product_sets']=c.execute("SELECT count(DISTINCT sha256(signature)) FROM (SELECT string_agg(predicate||':'||object_id,',' ORDER BY predicate,object_id) signature FROM edges WHERE predicate IN ('has_reactant','has_product') GROUP BY subject_id) t").fetchone()[0]
    
    report['reaction_extraction_flags']=dict(zip(['records_with_incomplete_roles','records_with_self_roles'],
      c.execute('SELECT count(*) FILTER (WHERE incomplete_role_rows>0), count(*) FILTER (WHERE self_role_rows>0) FROM reactions').fetchone()))
    print('[check] Checking source input checksums',flush=True)
    ingestion=json.loads((ROOT/'audit/ingestion_manifest.json').read_text())
    sources=[(x['source'],x['source_sha256']) for x in ingestion['jobs'] if 'source_sha256' in x]
    sources.append((ingestion['source_archive'],ingestion['archive_sha256']))
    available_sources=[item for item in sources if Path(item[0]).is_file()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        hashes=list(pool.map(lambda item:(item[0],digest(item[0])==item[1]),available_sources))
    assert all(ok for _,ok in hashes),hashes
    report['source_inputs_verified']=True if len(available_sources)==len(sources) else None
    report['source_files_unavailable']=len(sources)-len(available_sources)
    report['source_checksums_verified']=len(hashes)
    release_manifest=ROOT/'audit/parquet_manifest.json'
    if release_manifest.exists():
        previous=json.loads(release_manifest.read_text())
        raw_inputs=[item for item in previous['files'] if item['path'].startswith(
            ('raw_parquet/uspto/','raw_parquet/materials/','raw_parquet/structures/'))]
        assert all(digest(ROOT/item['path'])==item['sha256'] for item in raw_inputs)
        report['source_parquet_checksums_verified']=len(raw_inputs)
    report['files']={str(path.relative_to(ROOT)):{'rows':pq.ParquetFile(path).metadata.num_rows,
                     'bytes':path.stat().st_size,'sha256':digest(path),'schema':str(pq.read_schema(path))}
                     for path in sorted((ROOT/'graph_parquet').glob('*.parquet'))}
    report['elapsed_seconds']=round(time.time()-start,2)
    report['status']='PASS'
    (ROOT/'audit/graph_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('[PASS] Graph integrity, minimum scale, source links and source immutability verified',flush=True)

if __name__=='__main__':
    main()
