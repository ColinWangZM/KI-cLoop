from pathlib import Path
import os
from hashlib import sha256
import csv
import json
import math
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(os.environ.get("KICLOOP_PROPERTY_WORK", Path(__file__).resolve().parents[1] / "property_work")).resolve()
BASE=Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
import duckdb

QUANTITIES=[('molar_mass', 'Molar mass', 'Molar mass', 'M', 'g mol^-1', 'structure-derived property'), ('density', 'Mass density', 'Mass density', 'rho', 'kg m^-3', 'material or mixture property'), ('cell_volume', 'Unit-cell volume', 'Unit-cell volume', 'V_cell', 'm^3', 'crystal-structure property'), ('band_gap', 'Band-gap energy', 'Band-gap energy', 'E_g', 'eV', 'electronic-structure property'), ('relative_permittivity', 'Relative permittivity', 'Relative permittivity', 'epsilon_r', '1', 'material response; specify phase and frequency'), ('activation_energy', 'Apparent activation energy', 'Apparent activation energy', 'E_a', 'kJ mol^-1', 'reaction-model parameter'), ('activation_temperature', 'Activation energy divided by gas constant', 'Activation energy divided by gas constant', 'E_a/R_g', 'K', 'reaction-model parameter'), ('preexponential_factor', 'Arrhenius pre-exponential factor', 'Arrhenius pre-exponential factor', 'A', 'depends on rate-law order', 'reaction-model parameter'), ('log_preexponential', 'Logarithm of the numerical pre-exponential coefficient', 'Logarithm of the numerical pre-exponential coefficient', 'ln(A/A_unit)', '1', 'unit-convention-dependent model coordinate'), ('rate_coefficient', 'Rate coefficient', 'Rate coefficient', 'k', 'depends on rate-law order', 'reaction and temperature dependent'), ('reaction_order', 'Partial reaction order', 'Partial reaction order', 'n_i', '1', 'specific species and reaction-model parameter'), ('concentration', 'Amount concentration', 'Amount concentration', 'c_i', 'mol L^-1', 'species and condition dependent'), ('temperature', 'Thermodynamic temperature', 'Thermodynamic temperature', 'T', 'K', 'condition'), ('reaction_rate', 'Reaction rate', 'Reaction rate', 'r', 'mol L^-1 s^-1', 'reaction and condition dependent'), ('molar_gibbs_energy', 'Standard molar Gibbs energy', 'Standard molar Gibbs energy', 'G_m^0', 'kJ mol^-1', 'thermodynamic quantity; standard state required'), ('molar_enthalpy', 'Standard molar enthalpy', 'Standard molar enthalpy', 'H_m^0', 'kJ mol^-1', 'thermodynamic quantity; standard state required'), ('molar_entropy', 'Standard molar entropy', 'Standard molar entropy', 'S_m^0', 'kJ mol^-1 K^-1', 'thermodynamic quantity; standard state required'), ('reaction_gibbs_energy', 'Reaction Gibbs energy', 'Reaction Gibbs energy', 'Delta_r G', 'kJ mol^-1', 'reaction quantity; specified states required'), ('reaction_enthalpy', 'Reaction enthalpy', 'Reaction enthalpy', 'Delta_r H', 'kJ mol^-1', 'reaction quantity; specified states required'), ('reaction_entropy', 'Reaction entropy', 'Reaction entropy', 'Delta_r S', 'kJ mol^-1 K^-1', 'reaction quantity; specified states required'), ('volumetric_heat_source', 'Volumetric reaction heat source', 'Volumetric reaction heat source', 'q_v', 'W m^-3', 'reaction-network and condition dependent'), ('heat_transfer_coefficient', 'Wall heat-transfer coefficient', 'Wall heat-transfer coefficient', 'h', 'W m^-2 K^-1', 'engineering-model input'), ('thermal_conductivity', 'Thermal conductivity', 'Thermal conductivity', 'lambda', 'W m^-1 K^-1', 'material or mixture property'), ('tube_radius', 'Tube radius', 'Tube radius', 'R_tube', 'm', 'geometry'), ('temperature_rise', 'Temperature rise', 'Temperature rise', 'Delta T', 'K', 'engineering-model output'), ('heat_capacity', 'Specific heat capacity', 'Specific heat capacity', 'C_p', 'J kg^-1 K^-1', 'material or mixture property')]
Q={key:'quantity:'+key for key,*_ in QUANTITIES}
GOLD='https://goldbook.iupac.org/terms/view/'
FORMULAS=[
 dict(formula_id='arrhenius',equation='k(T) = A exp[-(Ea/Rg)/T]',
      inputs=['preexponential_factor','activation_temperature','temperature'],outputs=['rate_coefficient'],
      assumptions='T in kelvin; empirical A and Ea fixed within the specified fitted model; coefficient units depend on total order.',
      source=GOLD+'A00446'),
 dict(formula_id='power_law',equation='r = k(T) product_i c_i^n_i',
      inputs=['rate_coefficient','concentration','reaction_order'],outputs=['reaction_rate'],
      assumptions='Specified fitted reaction and species; concentrations in mol/L and rate in mol/L/s. Partial orders are empirical, not elementary stoichiometric coefficients.',
      source=GOLD+'O04322'),
 dict(formula_id='molar_gibbs_identity',equation='G_m^0 = H_m^0 - T S_m^0',
      inputs=['molar_enthalpy','temperature','molar_entropy'],outputs=['molar_gibbs_energy'],
      assumptions='All molar quantities at the same temperature, standard state and phase. This is not an activation barrier equation.',
      source=GOLD+'G02629'),
 dict(formula_id='reaction_gibbs_identity',equation='Delta_r G = Delta_r H - T Delta_r S',
      inputs=['reaction_enthalpy','temperature','reaction_entropy'],outputs=['reaction_gibbs_energy'],
      assumptions='Same defined reaction, temperature and thermodynamic states. Does not identify Ea or determine the kinetic rate.',
      source=GOLD+'G02629'),
 dict(formula_id='reaction_standard_gibbs_sum',equation='Delta_r G^0 = sum_i nu_i G_m,i^0',
      inputs=['molar_gibbs_energy'],outputs=['reaction_gibbs_energy'],
      assumptions='Requires a balanced reaction and every species standard-state molar Gibbs energy; positive nu for products. No numerical reaction Gibbs energies are assigned here.',
      source=GOLD+'G02629'),
 dict(formula_id='reaction_heat_source',equation='q_v[W/m3] = -1e6 sum_j Delta_r H_j[kJ/mol] r_j[mol/L/s]',
      inputs=['reaction_enthalpy','reaction_rate'],outputs=['volumetric_heat_source'],
      assumptions='Signed reaction enthalpies for the included reaction network; factor 1e6 converts kJ/L/s to W/m3.',
      source='general reaction energy balance'),
 dict(formula_id='cylindrical_screening',equation='Delta T = q_v [R_tube/(2h) + R_tube^2/(4lambda)]',
      inputs=['volumetric_heat_source','tube_radius','heat_transfer_coefficient','thermal_conductivity'],outputs=['temperature_rise'],
      assumptions='Steady cylindrical radial conduction with uniform volumetric heat source, constant conductivity and convective outer boundary.',
      source='analytical cylindrical heat balance'),
]
SCHEMA=pa.schema([
 ('property_value_id',pa.string()),('edge_id',pa.string()),('subject_id',pa.string()),('quantity_id',pa.string()),
 ('value',pa.float64()),('unit',pa.string()),('value_raw',pa.string()),('source_unit',pa.string()),
 ('source_ref',pa.string()),('source_sha256',pa.string()),('source_row',pa.int64()),('source_field',pa.string()),
 ('context_id',pa.string()),('context_json',pa.string()),('species_label',pa.string()),
 ('ci95_low',pa.float64()),('ci95_high',pa.float64()),('evidence_status',pa.string()),('unit_basis',pa.string())])

def j(value): return json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True)
def hash_file(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def edgeid(s,p,o):return 'edge:'+sha256(j([s,p,o]).encode()).hexdigest()

def main():
    start=time.time()
    out=ROOT/'graph_parquet'
    out.mkdir(parents=True, exist_ok=True)
    c=duckdb.connect()
    c.execute('SET threads=24')
    c.execute("SET memory_limit='40GB'")
    models=json.loads((ROOT/'inputs/closed_loop_models.json').read_text())
    species=json.loads((ROOT/'inputs/project_species.json').read_text())
    source_hash={f.name:hash_file(f) for f in (ROOT/'inputs').iterdir() if f.is_file()}
    param_rows={row['route']:row for row in csv.DictReader((ROOT/'inputs/kinetic_parameter_summary.csv').open())}
    for route,model in models.items():
        for key,value in model['parameters'].items():
            assert math.isclose(float(param_rows[route][key]),value,rel_tol=1e-12,abs_tol=1e-12),(route,key)
    base_hashes={name:hash_file(BASE/'graph_parquet'/name) for name in ['nodes.parquet','edges.parquet']}
    nodes=[];edges=[];values=[];evidence=[];model_rows=[]

    def node(node_id,node_type,label,source_dataset,source_key):
        nodes.append(dict(node_id=node_id,node_type=node_type,label=label,source_dataset=source_dataset,source_key=source_key))

    def relation(s,p,o,status,context=None,formula_id=None,source_ref=None,notes=None):
        eid=edgeid(s,p,o)
        edges.append(dict(edge_id=eid,subject_id=s,predicate=p,object_id=o,evidence_status=status,
                          evidence_table='property_values.parquet' if p=='has_property' else 'quantity_relation_evidence.parquet'))
        if p!='has_property':
            evidence.append(dict(edge_id=eid,subject_id=s,predicate=p,object_id=o,evidence_status=status,
                                 context_id=context,formula_id=formula_id,source_ref=source_ref,notes=notes))
        return eid

    def prop(subject,key,value,unit,source_ref,source_field,context_id,context=None,species_label=None,
             status='source_value',raw=None,source_unit=None,row=None,ci=None):
        assert key in Q and math.isfinite(float(value))
        eid=relation(subject,'has_property',Q[key],'contextual_property_values')
        vkey=[subject,key,source_ref,source_field,context_id,species_label,row]
        values.append(dict(property_value_id='property:'+sha256(j(vkey).encode()).hexdigest(),
            edge_id=eid,subject_id=subject,quantity_id=Q[key],value=float(value),unit=unit,
            value_raw=str(value) if raw is None else str(raw),source_unit=source_unit or unit,
            source_ref=source_ref,source_sha256=source_hash.get(Path(source_ref).name),source_row=row,
            source_field=source_field,context_id=context_id,context_json=j(context or {}),species_label=species_label,
            ci95_low=ci[0] if ci else None,ci95_high=ci[1] if ci else None,evidence_status=status,
            unit_basis='project units or declared calculation convention'))

    for key,label,cn,symbol,unit,scope in QUANTITIES:
        node(Q[key],'PhysicalQuantity',label,'physical_quantity_schema',key)
    for ds,label in [('project_kinetics','Project kinetic fit'),('project_comsol','Project COMSOL model')]:
        node('dataset:'+ds,'Dataset',label,'project_source_registry',ds)
    for name,row in species.items():
        if not row['in_existing_graph']:
            node(row['node_id'],'Molecule',row['label'],'project_kinetics',row['canonical_inchi'])
            for element in row['elements']:
                relation(row['node_id'],'has_element','element:'+element,'validated_project_structure',
                         source_ref='inputs/project_species.json',notes='Explicitly declared project catalyst structure, validated by RDKit; not an experimental USPTO observation.')
        prop(row['node_id'],'molar_mass',row['molar_mass_g_mol'],'g mol^-1','inputs/project_species.json',name,
             'canonical_structure',context={'rdkit_version':row['rdkit_version'],'method':row['molar_mass_method']},
             species_label=name,status='computed_from_structure_not_measured')

    branches={
      'tbhp_bzcl':[(1,[('H2O2','order_H2O2'),('TBCL','order_TBCL')],None,'TBHP')],
      'tbhp_wpo4':[(1,[('H2O2','order_H2O2'),('TBA','order_TBA_main')],'cat_order_1','TBHP'),
                    (2,[('TBA','order_TBA_side'),('TBHP','order_TBHP_side')],'cat_order_2','DTBP')],
      'tbhp_cf3':[(1,[('H2O2','order_H2O2'),('TBA','order_TBA_main')],'cat_order_1','TBHP'),
                  (2,[('TBA','order_TBA_side'),('TBHP','order_TBHP_side')],'cat_order_2','DTBP')],
      'tbpb_benzaldehyde':[(1,[('Benzaldehyde','order_benzaldehyde'),('TBHP','order_TBHP')],'cat_order_1','TBPB')],
    }
    for route,bs in branches.items():
        model=models[route];pars=model['parameters'];meta=model['metadata']
        ordered=sorted(model['conditions'],key=lambda k:(model['conditions'][k][0]['temperature_C'],k))
        anchor_name=ordered[len(ordered)//2]
        
        anchor=model['conditions'][anchor_name][1]
        T=anchor['temperature_C']+273.15
        for number,reactants,cat_key,product in bs:
            mid=f'model:kinetics:{route}:r{number}'
            orders=reactants+([(meta['catalyst'],cat_key)] if cat_key else [])
            total_order=sum(pars[key] for _,key in orders)
            coefficient_unit=f'(mol L^-1)^({1-total_order:.12g}) s^-1'
            context={'route':route,'branch':number,'catalyst':meta['catalyst'],'solvent':meta['solvent'],
                     'calibration_domain':model['domain'],'domain_note':model['domain_note'],
                     'interpretation':model['interpretation'],'fit_method':model['method'],
                     'concentration_unit':'mol L^-1','rate_unit':'mol L^-1 s^-1'}
            node(mid,'ReactionModel',f"{meta['label']} reaction {number}: {product}",'project_kinetics',f'{route}/reaction_{number}')
            relation(mid,'recorded_in','dataset:project_kinetics','project_source',context=mid,
                     source_ref='inputs/closed_loop_models.json')
            for name,_ in reactants:
                relation(mid,'has_reactant',species[name]['node_id'],'project_reaction_model',context=mid,
                         source_ref='inputs/closed_loop_models.json',notes='Species included in fitted reactant model; not a complete atom-balanced equation.')
            relation(mid,'has_product',species[product]['node_id'],'project_reaction_model',context=mid,
                     source_ref='inputs/closed_loop_models.json')
            for q,key,unit in [('activation_energy',f'Ea{number}_kJ_mol','kJ mol^-1'),
                               ('activation_temperature',f'Ea{number}_over_R','K'),
                               ('log_preexponential',f'lnA{number}','1')]:
                ci=model['uncertainty']['intervals'].get(key)
                prop(mid,q,pars[key],unit,'inputs/closed_loop_models.json',f'{route}.parameters.{key}',mid,
                     context=context,status='fitted_apparent_parameter',ci=(ci['low'],ci['high']) if ci else None)
            for name,key in orders:
                ci=model['uncertainty']['intervals'].get(key)
                prop(mid,'reaction_order',pars[key],'1','inputs/closed_loop_models.json',f'{route}.parameters.{key}',mid,
                     context=context,species_label=name,status='fitted_apparent_parameter',ci=(ci['low'],ci['high']) if ci else None)
                if name in species:
                    relation(species[name]['node_id'],'influences',Q['reaction_rate'],'conditional_fitted_model_dependence',
                             context=mid,formula_id='power_law',source_ref='inputs/closed_loop_models.json',
                             notes=j({'via':'concentration','partial_order':pars[key],'held_fixed':'temperature and other concentrations','not_universal_species_causality':True}))
            A=math.exp(pars[f'lnA{number}'])
            k=math.exp(pars[f'lnA{number}']-pars[f'Ea{number}_over_R']/T)
            rate=k
            for name,key in reactants:rate*=max(anchor[name],0)**pars[key]
            if cat_key:rate*=anchor['catalyst']**pars[cat_key]
            point_context={**context,'anchor_condition':anchor_name,'temperature_K':T,'time_s':anchor['time'],
                           'concentrations_mol_L':{name:anchor[name] for name,_ in reactants},
                           'catalyst_mol_L':anchor['catalyst'] if cat_key else None,
                           'uncertainty':'point calculation only; no propagated confidence interval',
                           'concentration_basis':'processed workbook state, not all independently measured'}
            prop(mid,'preexponential_factor',A,coefficient_unit,'inputs/closed_loop_models.json',f'{route}.parameters.lnA{number}',
                 mid,context=context,status='derived_from_fitted_lnA')
            for q,value,unit in [('rate_coefficient',k,coefficient_unit),('temperature',T,'K'),('reaction_rate',rate,'mol L^-1 s^-1')]:
                prop(mid,q,value,unit,'inputs/closed_loop_models.json',f'{route}.conditions.{anchor_name}[1]',
                     mid+':anchor',context=point_context,status='derived_model_point' if q!='temperature' else 'source_condition')
            for name,_ in reactants:
                prop(species[name]['node_id'],'concentration',anchor[name],'mol L^-1','inputs/closed_loop_models.json',
                     f'{route}.conditions.{anchor_name}[1].{name}',mid+':anchor',context=point_context,
                     species_label=name,status='processed_concentration_state')
            relation(Q['reaction_order'],'related',Q['reaction_rate'],'specified_rate_law_association',context=mid,
                     formula_id='power_law',source_ref='inputs/closed_loop_models.json',
                     notes='The species-specific exponent is part of this empirical rate law; this does not imply that increasing an exponent always increases the rate.')
            model_rows.append(dict(node_id=mid,route=route,branch=number,metadata_json=j(context),
                                   parameters_json=j({key:pars[key] for _,key in orders}|{f'lnA{number}':pars[f'lnA{number}'],f'Ea{number}_over_R':pars[f'Ea{number}_over_R']}),
                                   anchor_json=j(point_context),source_ref='inputs/closed_loop_models.json'))

    eng='model:engineering:comsol:path2'
    node(eng,'EngineeringModel','Axisymmetric tubular reactor with coupled flow, species transport and heat transfer','project_comsol','Path2')
    relation(eng,'recorded_in','dataset:project_comsol','model_description',context=eng,
             source_ref='knowledge/graph/reactor_evidence.jsonl',
             notes='Reaction chemistry couples species transport and heat release to the flow and temperature fields.')
    for key in ['density','heat_capacity']:
        relation(eng,'related',Q[key],'model_description',context=eng,
                 source_ref='knowledge/graph/reactor_evidence.jsonl',
                 notes='Mixture property involved in coupled flow and heat transfer.')

    for formula in FORMULAS:
        for src in formula['inputs']:
            for dst in formula['outputs']:
                relation(Q[src],'derive',Q[dst],'formula_dependency',context='formula:'+formula['formula_id'],
                         formula_id=formula['formula_id'],source_ref=formula['source'],
                         notes='Input participates in the complete expression; this edge alone is not sufficient to calculate the output. '+formula['assumptions'])
    for src,dst,formula,sign,scope in [
      ('temperature','rate_coefficient','arrhenius','positive for fitted positive Ea','fixed A and Ea'),
      ('activation_energy','rate_coefficient','arrhenius','negative','fixed A and T; does not compare independently fitted reactions'),
      ('concentration','reaction_rate','power_law','depends on the fitted partial order','same reaction, fixed temperature and other concentrations'),
      ('volumetric_heat_source','temperature_rise','cylindrical_screening','positive','fixed radius, h and thermal conductivity'),
      ('tube_radius','temperature_rise','cylindrical_screening','positive','fixed positive heat source, h and thermal conductivity'),
      ('heat_transfer_coefficient','temperature_rise','cylindrical_screening','negative','fixed positive heat source, radius and thermal conductivity'),
      ('thermal_conductivity','temperature_rise','cylindrical_screening','negative','fixed positive heat source, radius and h')]:
        relation(Q[src],'influences',Q[dst],'conditional_equation_dependence',context='formula:'+formula,
                 formula_id=formula,source_ref=next(f['source'] for f in FORMULAS if f['formula_id']==formula),
                 notes=j({'direction':sign,'held_fixed':scope}))
    
    relation(Q['activation_energy'],'derive',Q['rate_coefficient'],'formula_dependency',context='formula:arrhenius',
             formula_id='arrhenius',source_ref=GOLD+'A00446',notes='Convert Ea with a dimensionally consistent gas constant; requires A and T as well.')

    print(f'[project] {len(nodes)} property nodes; {len(values)} contextual project property values',flush=True)
    pq.write_table(pa.Table.from_pylist(nodes),out/'property_nodes.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(values,schema=SCHEMA),out/'project_property_values.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(model_rows),out/'reaction_models.parquet',compression='zstd')
    quantity_rows=[dict(node_id=Q[key],key=key,label=label,alternative_label=cn,symbol=symbol,
                        preferred_unit=unit,scope=scope) for key,label,cn,symbol,unit,scope in QUANTITIES]
    pq.write_table(pa.Table.from_pylist(quantity_rows),out/'physical_quantities.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(FORMULAS),out/'quantity_formulas.parquet',compression='zstd')

    material_parts=[];material_audit=[]
    for field,key,unit,source_unit,factor,constraint in [
      ('density','density','kg m^-3','g cm^-3',1000.,'x > 0'),
      ('volume','cell_volume','m^3','angstrom^3',1e-30,'x > 0'),
      ('band_gap','band_gap','eV','eV',1.,'x >= 0'),
      ('e_total','relative_permittivity','1','1',1.,'x > 0')]:
        source=BASE/f'raw_parquet/materials/material_project__{field}.parquet'
        source_sha=hash_file(source)
        
        input_copy=ROOT/'inputs'/source.name
        if not input_copy.exists(): input_copy.write_bytes(source.read_bytes())
        assert hash_file(input_copy)==source_sha
        input_rows,valid_rows=c.execute(f"SELECT count(*),count(*) FILTER (WHERE isfinite(x) AND {constraint}) FROM (SELECT try_cast(value_raw AS DOUBLE) x FROM read_parquet(?))",[str(source)]).fetchone()
        material_audit.append(dict(field=field,input_rows=input_rows,valid_rows=valid_rows,
                                   omitted_non_numeric_or_invalid=input_rows-valid_rows,source_sha256=source_sha))
        material_parts.append(f"""SELECT
          'property:'||sha256(to_json(list_value(dataset,record_id,property_name,source_member,cast(source_row AS VARCHAR)))) property_value_id,
          'edge:'||sha256(to_json(list_value('material:'||dataset||':'||record_id,'has_property','{Q[key]}'))) edge_id,
          'material:'||dataset||':'||record_id subject_id,'{Q[key]}' quantity_id,
          x*{factor} AS "value",'{unit}' unit,value_raw,'{source_unit}' source_unit,
          'inputs/{input_copy.name}' source_ref,'{source_sha}' source_sha256,source_row,
          property_name source_field,dataset||':'||record_id context_id,
          to_json(struct_pack(source_member:=source_member,source_kind:='Materials Project calculated property',unit_mapping:='official emmet schema',specific_computational_details:='retain source record; not inferred from formula')) context_json,
          NULL::VARCHAR species_label,NULL::DOUBLE ci95_low,NULL::DOUBLE ci95_high,
          'source_material_calculation' evidence_status,'Materials Project emmet schema' unit_basis
          FROM (SELECT *,try_cast(value_raw AS DOUBLE) x FROM read_parquet('{input_copy}'))
          WHERE isfinite(x) AND {constraint}""")
    query=' UNION ALL '.join(material_parts)
    c.execute(f"COPY ({query}) TO '{out}/material_property_values.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)")
    c.execute(f"COPY (SELECT * FROM read_parquet('{out}/material_property_values.parquet') UNION ALL SELECT * FROM read_parquet('{out}/project_property_values.parquet')) TO '{out}/property_values.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)")
    
    edges=[row for row in edges if row['predicate']!='has_property']
    pq.write_table(pa.Table.from_pylist(edges),out/'quantity_relation_edges.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(evidence),out/'quantity_relation_evidence.parquet',compression='zstd')
    c.execute(f"""COPY (SELECT DISTINCT edge_id,subject_id,'has_property' predicate,quantity_id object_id,
      'contextual_property_values' evidence_status,'property_values.parquet' evidence_table
      FROM read_parquet('{out}/property_values.parquet')
      UNION SELECT * FROM read_parquet('{out}/quantity_relation_edges.parquet'))
      TO '{out}/property_edges.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    
    c.execute(f"COPY (SELECT * FROM read_parquet('{BASE}/graph_parquet/nodes.parquet') UNION ALL SELECT * FROM read_parquet('{out}/property_nodes.parquet') ORDER BY node_id) TO '{out}/nodes.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)")
    source_evidence_prefix = (Path(os.path.relpath(BASE / 'graph_parquet', out)).as_posix() + '/').replace("'", "''")
    c.execute(f"""COPY (SELECT edge_id,subject_id,predicate,object_id,evidence_status,
        '{source_evidence_prefix}'||evidence_table evidence_table
        FROM read_parquet('{BASE}/graph_parquet/edges.parquet')
        UNION ALL SELECT * FROM read_parquet('{out}/property_edges.parquet')
        ORDER BY subject_id,predicate,object_id) TO '{out}/edges.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    (ROOT/'audit/material_property_selection.json').write_text(json.dumps(material_audit,indent=2)+'\n')
    (ROOT/'audit/base_hashes_before.json').write_text(json.dumps(base_hashes,indent=2)+'\n')
    (ROOT/'audit/build_summary.json').write_text(json.dumps({'base_export':str(BASE),'property_node_count':len(nodes),
        'project_property_value_count':len(values),'material_property_value_count':sum(x['valid_rows'] for x in material_audit),
        'formula_count':len(FORMULAS),'elapsed_seconds':round(time.time()-start,2)},indent=2)+'\n')
    print(f'[complete] Graph and property tables saved; elapsed {time.time()-start:.1f}s',flush=True)

if __name__=='__main__':main()
