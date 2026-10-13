import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'kinetics'))
from kinetic_core import load_data, predict, physical_to_scaled, _rates
from screen import objectives, candidate_dataset
from case_facts import scientific_facts, GROUP_SYSTEM


def scenario(data, initial, temperatures, catalyst, times):
    n=len(temperatures)
    return replace(data, initial=np.tile(initial,(n,1)),conditions=np.c_[np.asarray(temperatures)+273.15,np.full(n,catalyst)],times=np.tile(times,(n,1)))


def close(actual, expected, label, atol=1e-7):
    if not np.allclose(actual,expected,rtol=1e-6,atol=atol):
        raise ValueError(f'Numerical case check failed: {label}')


def verify_calculation(c, results, bootstrap):
    route=c.get('route')
    tool=c['tool']
    if tool in ('route_evaluation','literature_condition_transfer'):
        return {'tool':tool,'status':'tabulated evidence used by deterministic renderer'}
    if tool == 'thermal_scale_analysis':
        required = {'rate_constant', 'reaction_rate', 'heat_source', 'concentration_dynamics', 'peak_heat_source', 'temperature_margin', 'thermal_resistance', 'radius', 'diameter', 'volumetric_flow', 'diameter_ratio', 'flow_ratio', 'product_mass_flow'}
        if set(c['equations']) != required or not c['symbols'] or not c['assumptions']:
            raise ValueError('Incomplete thermal-analysis formulation.')
        result = c['results']
        records = result.get('scenarios', [result])
        for item in records:
            radius = item['estimated_radius_mm']
            diameter = item['estimated_diameter_mm']
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError('The reported analytical radius must be positive and finite.')
            close(diameter, 2 * radius, route + ' analytical diameter')
        if 'scenarios' in result:
            radii = [item['estimated_radius_mm'] for item in records]
            close(result['radius_range_mm'], [min(radii), max(radii)], route + ' analytical size range')
            if result['representative_scenario'] not in {item['scenario'] for item in records}:
                raise ValueError('The representative thermal scenario must identify a result record.')
        return {'tool': tool, 'status': 'Formula structure and reported-result consistency verified'}
    data=load_data(route)
    summary=json.loads((results/route/'final_summary.json').read_text())
    q=np.asarray(summary['q'])
    if tool=='temperature_comparison':
        initial=[c['initial_mol_L'][s] for s in data.species]
        sd=scenario(data,initial,c['temperatures_C'],c['catalyst_mol_L'],np.linspace(0,c['time_s'],201))
        pred=predict(sd,q,.25)
        values=objectives(data,sd.initial,pred[:,-1])
        for j,expected in enumerate(c['predictions']):
            close(pred[j,-1],[expected['final_mol_L'][s] for s in data.species],route+' endpoint')
            close(values['yield'][j],expected['yield_fraction'],route+' yield')
        if bootstrap:
            rows=[]
            for _,sample in pd.read_csv(results/route/'bootstrap_parameters.csv').iterrows():
                if sample['converged']:
                    bp=predict(sd,physical_to_scaled(data,sample.to_dict()),.25)
                    rows.append(objectives(data,sd.initial,bp[:,-1])['yield'])
            rows=np.asarray(rows)
            for j,key in enumerate(['low_temperature_yield_95pct','high_temperature_yield_95pct']):
                interval=c['bootstrap'][key]
                close(np.quantile(rows[:,j],[.025,.5,.975]),[interval[k] for k in ['low','median','high']],route+' bootstrap',atol=2e-5)
        return {'tool':tool,'status':'endpoint integration and parameter bootstrap verified' if bootstrap else 'endpoint integration verified'}
    if tool in ('recommend_conditions','structural_transfer'):
        condition=c['recommendation'] if tool=='recommend_conditions' else c['reference_screened_condition']
        sd=candidate_dataset(data,pd.DataFrame([condition]),np.array([0.,condition['time']]))
        if tool=='recommend_conditions':
            values=objectives(data,sd.initial[0],predict(sd,q,.25)[0,-1])
            for name in ['yield','selectivity','product','score']:
                close(values[name],condition[name],route+' recommendation '+name)
        else:
            for item in c['borrowed_kinetics_scenarios']['cases']:
                trial=q.copy()
                trial[::5] += math.log(item['assumed_rate_multiplier'])
                values=objectives(data,sd.initial[0],predict(sd,trial,.25)[0,-1])
                close(values['yield'],item['yield_fraction'],'borrowed-rate yield')
        return {'tool':tool,'status':'conditional ODE calculations verified'}
    raise ValueError('Unknown calculation tool: '+tool)


def validate_groups(groups,facts):
    ids=[f['id'] for f in facts]
    flat=[k for group in groups for k in group]
    if not 3<=len(groups)<=5 or not all(groups) or len(flat)!=len(set(flat)) or set(flat)!=set(ids):
        raise ValueError('Every scientific statement must appear exactly once in 3 to 5 nonempty groups.')


def main():
    p=argparse.ArgumentParser(description='Verify numerical case calculations and reproduce immutable-fact answers.')
    p.add_argument('--results',type=Path,default=ROOT/'kinetics/results')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/cases')
    p.add_argument('--skip-bootstrap',action='store_true')
    p.add_argument('--regroup',action='store_true')
    p.add_argument('--model',default=os.environ.get('QWEN_MODEL_PATH'))
    args=p.parse_args()
    generator=None
    if args.regroup:
        if not args.model:
            p.error('--regroup requires --model or QWEN_MODEL_PATH.')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype='auto',device_map='auto',local_files_only=True).eval()
        def generator(messages):
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(text,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=256,do_sample=False)
            return tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    args.output.mkdir(parents=True,exist_ok=True)
    checks=[]
    for file in sorted((HERE/'data/cases').glob('*.json')):
        case=json.loads(file.read_text())
        for c in case['calculations']:
            checks.append(dict(case=case['case'],**verify_calculation(c,args.results,not args.skip_bootstrap)))
        facts=scientific_facts(case['calculations'])
        groups=case['groups']
        engineering = all(c['tool'] == 'thermal_scale_analysis' for c in case['calculations'])
        generation = 'deterministic scientific-fact grouping' if engineering else 'recorded Qwen grouping'
        if generator and not engineering:
            messages=[dict(role='system',content=GROUP_SYSTEM),dict(role='user',content=json.dumps(dict(question=case['question'],facts=facts),ensure_ascii=False))]
            raw=generator(messages)
            groups=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip()))['groups']
            generation='local Qwen grouping'
        validate_groups(groups,facts)
        lookup={f['id']:f['text'] for f in facts}
        paragraphs=[' '.join(lookup[k] for k in group) for group in groups]
        answer='Case '+str(case['number'])+': '+case['question']+'\n\n'+'\n\n'.join(paragraphs)+'\n'
        if engineering:
            calculation = case['calculations'][0]
            answer += '\nEquations\n' + '\n'.join(calculation['equations'].values()) + '\n'
            answer += '\nSymbols\n' + '\n'.join(f'{symbol}: {meaning}' for symbol, meaning in calculation['symbols'].items()) + '\n'
            answer += '\nAnalytical results\n' + json.dumps(calculation['results'], indent=2) + '\n'
        (args.output/(case['case']+'.txt')).write_text(answer)
        rendered = dict(case=case['case'],question=case['question'],scientific_facts=facts,groups=groups,generation=generation)
        if engineering:
            rendered['calculations'] = case['calculations']
        (args.output/(case['case']+'.json')).write_text(json.dumps(rendered,ensure_ascii=False,indent=2)+'\n')
    (args.output/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(f'Processed {len(checks)} case records.')

if __name__=='__main__':main()
