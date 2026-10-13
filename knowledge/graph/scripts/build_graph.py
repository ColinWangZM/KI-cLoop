from pathlib import Path
import os
import json
import sys
import time

ROOT = Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

OUT = ROOT/'graph_parquet'
START = time.time()

def log(message):
    print(f'[{time.time()-START:6.1f}s] {message}', flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET threads=24")
    c.execute("SET memory_limit='48GB'")
    c.execute("SET temp_directory='"+str(ROOT/'tmp')+"'")

    def export(query, name):
        c.execute(f"COPY ({query}) TO '{OUT/name}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        log('Saved '+name)

    c.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{ROOT}/raw_parquet/uspto/*.parquet')")
    c.execute(f"CREATE VIEW identity_map AS SELECT * FROM read_parquet('{ROOT}/raw_parquet/molecule_identity_map.parquet')")
    c.execute(f"CREATE VIEW material_features AS SELECT * FROM read_parquet('{ROOT}/raw_parquet/material_features.parquet')")
    log('Deduplicating validated molecular identities')
    c.execute('''CREATE TABLE molecules AS
      SELECT node_id, min(canonical_inchi) canonical_inchi,
             min(canonical_smiles) canonical_smiles, min(formula) formula,
             min(source_name) source_name, first(elements) elements,
             first(class_ids) class_ids, count(*) source_identity_count
      FROM identity_map WHERE node_id IS NOT NULL GROUP BY node_id''')
    export('SELECT * FROM molecules ORDER BY node_id', 'molecules.parquet')
    export('SELECT * FROM material_features ORDER BY node_id', 'materials.parquet')

    log('Joining USPTO roles to canonical molecules; retaining every source row')
    c.execute('''CREATE TABLE mapped AS
      WITH normalized AS (
        SELECT source_file, source_row, source_year, "source_xml_filename" source_xml,
               trim("ID") patent_source_id,
               regexp_replace(trim("ID"), '^US0+', 'US') patent_id,
               trim("procedure_text") procedure_text_extracted,
               trim("reactant_inchi") reactant_inchi,
               trim("product_inchi") product_inchi,
               trim("solvent_inchi") solvent_inchi
        FROM raw
      )
      SELECT n.*, r.node_id reactant_id, p.node_id product_id, s.node_id solvent_id,
             'reaction:' || sha256(to_json(list_value(patent_id,procedure_text_extracted))) reaction_id,
             CASE
               WHEN NOT regexp_full_match(patent_id, 'US[0-9]+[A-Z0-9]*') OR patent_id IS NULL
                 THEN 'invalid_patent_id'
               WHEN procedure_text_extracted IS NULL OR length(procedure_text_extracted)=0
                 THEN 'missing_procedure'
               WHEN patent_id='US4192922' AND product_inchi=
                 'InChI=1S/C11H14O3/c1-11(2,3)14-13-10(12)9-7-5-4-6-8-9/h4-8H,1-3H3'
                 THEN 'reviewed_tbpb_is_polymerization_initiator_not_product'
               WHEN NOT regexp_matches(procedure_text_extracted,
                 '(Add|Mix|Dissolve|React|Heat|Stir|Cool|Synthesize|ApparatusAction):')
                 THEN 'no_preparative_action_detected'
               WHEN regexp_matches(lower(procedure_text_extracted),
                 '(obtained from commercial sources|purchased from|commercially available)')
                 AND NOT regexp_matches(procedure_text_extracted, '(React|Heat|Stir|Synthesize):')
                 THEN 'commercial_or_nonreaction_context'
               ELSE 'candidate'
             END row_screen
      FROM normalized n
      LEFT JOIN identity_map r ON n.reactant_inchi=r.inchi
      LEFT JOIN identity_map p ON n.product_inchi=p.inchi
      LEFT JOIN identity_map s ON n.solvent_inchi=s.inchi''')
    log('Grouping extracted experimental records by patent and exact procedure text')
    c.execute('''CREATE TABLE accepted_ids AS
      SELECT reaction_id FROM mapped WHERE row_screen='candidate'
      GROUP BY reaction_id
      HAVING bool_or(product_id IS NOT NULL AND reactant_id IS NOT NULL
                     AND product_id <> reactant_id)''')
    c.execute('''CREATE TABLE accepted_rows AS
      SELECT m.* FROM mapped m JOIN accepted_ids USING(reaction_id)''')
    c.execute('''CREATE TABLE reactions AS
      SELECT reaction_id node_id, patent_id, min(patent_source_id) patent_source_id,
             first(procedure_text_extracted) procedure_text_extracted,
             sha256(first(procedure_text_extracted)) procedure_sha256,
             count(*) source_row_count, count(DISTINCT product_id) product_count,
             count(DISTINCT reactant_id) reactant_count,
             count(DISTINCT solvent_id) solvent_count,
             count(*) FILTER (WHERE product_id IS NULL OR reactant_id IS NULL) incomplete_role_rows,
             count(*) FILTER (WHERE product_id=reactant_id) self_role_rows,
             'source_extracted_not_manually_verified' evidence_status
      FROM accepted_rows GROUP BY reaction_id,patent_id''')
    export('SELECT * FROM reactions ORDER BY node_id', 'reactions.parquet')
    export('''SELECT reaction_id,source_file,source_row,source_year,source_xml,
                     patent_source_id,reactant_id,product_id,solvent_id
              FROM accepted_rows ORDER BY source_file,source_row''', 'reaction_sources.parquet')
    export('''SELECT m.source_file,m.source_row,m.patent_source_id,m.reaction_id,
                CASE WHEN a.reaction_id IS NOT NULL THEN 'included'
                     WHEN m.row_screen='candidate' THEN 'no_distinct_valid_reactant_product_pair'
                     ELSE m.row_screen END disposition
              FROM mapped m LEFT JOIN accepted_ids a USING(reaction_id)
              ORDER BY source_file,source_row''', 'source_row_disposition.parquet')
    c.execute('''CREATE TABLE patents AS
        SELECT DISTINCT patent_id, 'patent:' || patent_id node_id FROM reactions''')
    export('SELECT * FROM patents ORDER BY node_id', 'patents.parquet')

    log('Creating composition and molecular structural classes')
    classes = json.loads((ROOT/'audit/molecule_validation.json').read_text())['rules']
    class_records = [{'node_id': r['class_id'], 'label':r['label'], 'rule':r['smarts'],
                      'rule_type':'rdkit_smarts', 'domain':'Molecule'} for r in classes]
    labels = {'elemental':'Elemental material', 'binary':'Binary material',
              'ternary':'Ternary material', 'four_or_more_elements':'Material with four or more elements',
              'oxygen_containing':'Oxygen-containing material', 'carbon_containing':'Carbon-containing material',
              'nitrogen_containing':'Nitrogen-containing material', 'halogen_containing':'Halogen-containing material',
              'qmof_framework_record':'QMOF framework-material record'}
    for key,label in labels.items():
        rule = {'elemental':'number of elements = 1', 'binary':'number of elements = 2',
                'ternary':'number of elements = 3', 'four_or_more_elements':'number of elements >= 4',
                'oxygen_containing':'O present in parsed formula', 'carbon_containing':'C present in parsed formula',
                'nitrogen_containing':'N present in parsed formula',
                'halogen_containing':'F, Cl, Br, I, At or Ts present in parsed formula',
                'qmof_framework_record':'source dataset is qmof'}[key]
        class_records.append({'node_id':'class:material:'+key,'label':label,'rule':rule,
                              'rule_type':'composition_or_dataset','domain':'MaterialRecord'})
    pq.write_table(pa.Table.from_pylist(class_records),OUT/'classes.parquet',compression='zstd')
    c.execute(f"CREATE VIEW classes AS SELECT * FROM read_parquet('{OUT}/classes.parquet')")
    c.execute('''CREATE TABLE elements AS SELECT DISTINCT element FROM
        (SELECT unnest(elements) element FROM molecules
         UNION ALL SELECT unnest(elements) element FROM material_features)''')
    c.execute('''CREATE TABLE datasets AS SELECT DISTINCT dataset FROM material_features
                 UNION ALL SELECT 'uspto_grants_1976_2016_local_cml_derivative' ''')
    export('''SELECT node_id, 'Molecule' node_type, coalesce(nullif(source_name,''),formula) AS "label",
                  'uspto_grants_1976_2016_local_cml_derivative' source_dataset, canonical_inchi source_key
              FROM molecules
              UNION ALL
              SELECT node_id,'MaterialRecord',formula_raw,dataset,record_id FROM material_features
              UNION ALL
              SELECT node_id,'ReactionRecord',patent_id || ' extracted reaction',
                  'uspto_grants_1976_2016_local_cml_derivative',procedure_sha256 FROM reactions
              UNION ALL SELECT node_id,'PatentDocument',patent_id,
                  'uspto_grants_1976_2016_local_cml_derivative',patent_id FROM patents
              UNION ALL SELECT node_id,'StructuralClass',label,'explicit_rules',rule FROM classes
              UNION ALL SELECT 'element:'||element,'ChemicalElement',element,'periodic_table',element FROM elements
              UNION ALL SELECT 'dataset:'||dataset,'Dataset',dataset,'source_registry',dataset FROM datasets
              ORDER BY node_id''', 'nodes.parquet')

    log('Creating unique directed relations and per-edge source support')
    c.execute('''CREATE TABLE role_support AS
      SELECT DISTINCT reaction_id subject_id, 'has_reactant' predicate, reactant_id object_id,
                      source_file,source_row FROM accepted_rows WHERE reactant_id IS NOT NULL
      UNION ALL
      SELECT DISTINCT reaction_id,'has_product',product_id,source_file,source_row
                      FROM accepted_rows WHERE product_id IS NOT NULL
      UNION ALL
      SELECT DISTINCT reaction_id,'has_solvent',solvent_id,source_file,source_row
                      FROM accepted_rows WHERE solvent_id IS NOT NULL''')
    export('''SELECT 'edge:'||sha256(to_json(list_value(subject_id,predicate,object_id))) edge_id,
                     subject_id,predicate,object_id,source_file,source_row
              FROM role_support ORDER BY source_file,source_row,predicate''', 'reaction_edge_evidence.parquet')
    c.execute('''CREATE TABLE edge_tuples AS
      SELECT subject_id,predicate,object_id,
             'source_extracted_not_manually_verified' evidence_status,
             'reaction_edge_evidence.parquet' evidence_table FROM role_support
      GROUP BY subject_id,predicate,object_id
      UNION ALL
      SELECT node_id,'reported_in','patent:'||patent_id,
             'source_identifier','reaction_sources.parquet' FROM reactions
      UNION ALL
      SELECT node_id,'belongs_to',unnest(class_ids),'structure_rule','classes.parquet' FROM molecules
      UNION ALL
      SELECT node_id,'belongs_to',unnest(class_ids),'composition_or_dataset_rule','classes.parquet' FROM material_features
      UNION ALL
      SELECT node_id,'has_element','element:'||unnest(elements),'validated_structure','molecules.parquet' FROM molecules
      UNION ALL
      SELECT node_id,'has_element','element:'||unnest(elements),'parsed_source_formula','materials.parquet' FROM material_features
      UNION ALL
      SELECT node_id,'recorded_in','dataset:'||dataset,'source_record','materials.parquet' FROM material_features
      UNION ALL
      SELECT node_id,'recorded_in','dataset:uspto_grants_1976_2016_local_cml_derivative',
             'source_record','patents.parquet' FROM patents''')
    export('''SELECT 'edge:'||sha256(to_json(list_value(subject_id,predicate,object_id))) edge_id,
                    subject_id,predicate,object_id,evidence_status,evidence_table
              FROM edge_tuples ORDER BY subject_id,predicate,object_id''', 'edges.parquet')
    log('Graph export complete; independent validation is required before release')

if __name__ == '__main__':
    main()
