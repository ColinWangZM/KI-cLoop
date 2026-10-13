META = {key: dict(label=f'Path {i + 1}', side=i in (1, 2)) for i, key in enumerate(['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3', 'tbpb_benzaldehyde'])}
LABELS = {}

def scientific_facts(calculations):
    facts = []

    def label(s):
        return LABELS.get(s, s)

    def add(text, c, role='explanation'):
        facts.append(dict(id='F' + str(len(facts) + 1), text=text, role=role, required=True, evidence_ids=c.get('evidence_ids', [])))

    def params(c):
        rows = c['parameter_insights']['reactions']
        return '; '.join((f"{('Main' if r['reaction'] == 1 else 'Side')} reaction: apparent Ea = {r['Ea_kJ_mol']:.3f} kJ/mol, ln A = {r['lnA']:.4f}, orders " + ', '.join((f'{label(s)} {o:.3f}' for s, o in r['orders'].items())) for r in rows)) + '. These parameters describe the calibrated system and do not establish an elementary mechanism.'
    for c in calculations:
        tool = c['tool']
        if tool == 'temperature_comparison' and 'predictions' in c:
            lo, hi = c['predictions']
            add(f"At the same feed composition, catalyst concentration and {c['time_s']:.2f} s, predicted yield/selectivity is {lo['yield_fraction']:.2%}/{lo['selectivity_fraction']:.2%} at {lo['temperature_C']:g} °C and {hi['yield_fraction']:.2%}/{hi['selectivity_fraction']:.2%} at {hi['temperature_C']:g} °C. Yield and selectivity must be compared separately.", c, 'comparison')
            add(params(c), c, 'parameters')
            fs = c['rate_interpretation']['fixed_composition_Arrhenius_factors']
            add('At fixed instantaneous concentrations, heating from the lower to the higher temperature multiplies ' + ' and '.join((f"the {('main' if j == 1 else 'side')} rate by {fs[f'reaction_{j}_high_over_low']:.3f}" for j in range(1, len(fs) + 1))) + '. These are rate multipliers, not final-yield multipliers.', c)
            if META[c['route']]['side']:
                e = c['rate_interpretation']['concentration_exponents']
                add(f"The concentration factor in the side/main rate ratio is catalyst^{e['catalyst']:.3f} × TBA^{e['TBA']:.3f} × TBHP^{e['TBHP']:.3f} × H2O2^{e['H2O2']:.3f}. Competition changes as feedstocks are consumed and TBHP accumulates. The integrated side-reaction consumption of TBHP is {lo['integrated_side_extent_mol_L']:.4f} and {hi['integrated_side_extent_mol_L']:.4f} mol/L at the lower and higher temperatures, respectively.", c)
                preferred = lo if lo['selectivity_fraction'] >= hi['selectivity_fraction'] else hi
                add(f"For this endpoint, {preferred['temperature_C']:g} °C gives higher predicted selectivity. Temperature and residence time should be screened together. At the higher temperature, the predicted TBHP maximum is at approximately {hi['peak_time_s']:.1f} s on a {hi['peak_resolution_s']:.1f}-s grid; longer residence does not necessarily increase target yield.", c, 'decision')
            else:
                r = c['parameter_insights']['reactions'][0]
                add('The fitted network contains only the target-product channel. Its approximately 100% predicted selectivity is a model assumption and does not establish absence of by-products. An isolated 10% concentration increase multiplies the rate by ' + ', '.join((f'{v:.3f} for {label(s)}' for s, v in r['ten_percent_increase_fixed_others'].items())) + '. Check feed and catalyst metering and residence time, then validate within the measured temperature range.', c, 'decision')
            b = c['bootstrap']
            if b.get('performed'):
                v = b['delta_yield_low_minus_high_95pct']
                add(f"Across {b['n']} converged parameter-bootstrap draws, the conditional 95% interval for lower-minus-higher-temperature yield is [{100 * v['low']:.2f}, {100 * v['high']:.2f}] percentage points. " + ('The interval includes zero, so the yield ordering is unresolved under this conditional uncertainty.' if v['low'] <= 0 <= v['high'] else 'The interval excludes zero, but the comparison remains conditional on the fitted model.') + ' Model-form and data-processing uncertainty are not included.', c, 'uncertainty')
            z = c.get('measured_model_comparison')
            if z:
                add(f"For paired processed experimental observations, the higher-minus-lower-temperature yield difference is {z['observed_high_minus_low_yield_pp']:.2f} percentage points, versus {z['predicted_high_minus_low_yield_pp']:.2f} percentage points predicted. These observations participated in fitting; this is a residual check, not held-out validation. " + ('The discrepancy requires checking the condition-specific time series; parameter intervals do not represent model discrepancy.' if abs(z['observed_high_minus_low_yield_pp'] - z['predicted_high_minus_low_yield_pp']) > 3 else 'Use this comparison to check consistency of the trend.'), c, 'measured_check')
            else:
                add('No complete matched pair of processed observations is available at this exact comparison time. The temperature comparison is a model prediction, not experimental validation.', c, 'boundary')
        elif tool == 'recommend_conditions':
            r = c['recommendation']
            b = c['best_observed_by_same_score']
            ci = c['conditional_parameter_intervals_95pct']['yield']
            ratio = 'TBHP/benzaldehyde' if c['route'] == 'tbpb_benzaldehyde' else 'H2O2/' + ('TBCl' if c['route'] == 'tbhp_bzcl' else 'TBA')
            cat = 'no added catalyst' if c['route'] == 'tbhp_bzcl' else f"catalyst {r['catalyst']:.4f} mol/L"
            add(f"The finite-search recommendation for {META[c['route']]['label']} is {r['temperature_C']:g} °C, {r['time']:.2f} s, {ratio} = {r['feed_ratio']:.4g}, and {cat}. Predicted yield is {r['yield']:.2%} and selectivity is {r['selectivity']:.2%}.", c, 'comparison')
            add(params(c), c, 'parameters')
            add(f"The processed experimental point selected by the same objective is at {b['temperature_C']:g} °C and {b['time']:.2f} s, with yield {b['yield']:.2%}. The condition-screening score is {b['score']:.4f} for that observation and {r['score']:.4f} for the recommended prediction; it differs from the five-index route score. The recommended-yield conditional 95% parameter interval is [{ci['low']:.2%}, {ci['high']:.2%}] and excludes uncertainty from selecting the optimum.", c, 'comparison')
            x = c['parameter_insights']['reactions'][0]
            add('At otherwise fixed conditions, a 10% concentration increase multiplies the main rate by ' + ', '.join((f'{v:.3f} for {label(s)}' for s, v in x['ten_percent_increase_fixed_others'].items())) + '. Feed volumes and concentrations are coupled in practice; these local rate responses are not final-yield gains.', c)
            if META[c['route']]['side']:
                add('Temperature, residence time and composition should jointly control target formation and side-reaction consumption. ' + ('The predicted recommendation scores below the best processed observation, so it is a candidate trade-off for verification, not a demonstrated yield improvement.' if c['predicted_minus_observed_score'] < 0 else 'Any predicted score advantage requires experimental confirmation.'), c, 'decision')
            else:
                add("Verify metering of the most sensitive feeds and test nearby temperatures and residence times. A single-channel model's 100% selectivity is not measured purity. A recommendation at a domain boundary does not justify further extrapolation.", c, 'decision')
        elif tool == 'thermal_scale_analysis':
            add('Evaluate the reaction-rate law k_j(T) = A_j exp(-E_a,j/(R_g T)) and r_j = k_j(T) C_cat^m_j product_i(C_i^alpha_ij). The instantaneous heat source is q_v = sum_j((-Delta_H_j) r_j), using consistent SI units. Integrate dC/dt = S r(T_limit, C(t)) from C(0) = C_0 and use q_peak = max_t sum_j((-Delta_H_j) r_j(T_limit, C(t))) as the heat-source envelope. Reaction enthalpy and mixture properties must match the reaction system.', c, 'method')
            add('For a cylindrical tube, Delta_T = T_limit - T_coolant = q_peak [R_t/(2h) + R_t^2/(4k_th)]. Solving for the positive radius gives R_t = sqrt[(k_th/h)^2 + 4k_th Delta_T/q_peak] - k_th/h, and D_t = 2R_t. The exterior-film and radial-conduction terms scale linearly and quadratically with radius, respectively.', c, 'formula')
            result = c['results']
            if c['route'] == 'tbhp_wpo4':
                add(f"The analytical Route 2 estimate gives a candidate radius of {result['estimated_radius_mm']:.2f} mm and diameter of {result['estimated_diameter_mm']:.2f} mm. These are results of the cylindrical heat-balance estimate, not a prescribed geometry for a coupled COMSOL calculation.", c, 'result')
                add('A larger heat load or weaker exterior heat transfer reduces the candidate size. Stronger exterior heat transfer increases it, but radial conduction continues to constrain the temperature rise. Confirm reaction heat and heat-transfer conditions before evaluating a candidate geometry with coupled transport equations.', c, 'decision')
            else:
                low, high = result['radius_range_mm']
                selected = next(s for s in result['scenarios'] if s['scenario'] == result['representative_scenario'])
                add(f"Across the {len(result['scenarios'])} thermal-property scenarios, the analytical Route 4 radii range from {low:.2f} to {high:.2f} mm. The representative scenario gives radius {selected['estimated_radius_mm']:.2f} mm and diameter {selected['estimated_diameter_mm']:.2f} mm. These are conditional size estimates, not validated reactor dimensions.", c, 'result')
                add('Route 4 uses its own fitted rate law. Higher assumed reaction heat decreases the estimated size, whereas greater thermal conductivity increases it. Measure the reaction enthalpy and mixture thermal conductivity before interpreting these estimates as operating choices. A low apparent activation energy alone does not establish safety because concentration sensitivity can amplify heat-release errors.', c, 'decision')
            add('At unchanged mean velocity, Q/Q_ref = (R_t/R_ref)^2 and D_t/D_ref = R_t/R_ref. Product mass flow is m_dot_product = M_product C_product,out Q. A diameter ratio or volumetric-flow ratio does not by itself determine product throughput, since outlet concentration can depend on reaction and transport.', c, 'interpretation')
            add(' '.join(c['assumptions']), c, 'boundary')
        elif tool == 'structural_transfer':
            x = c['parameter_insights']['reactions'][0]
            add('The target reaction is p-tolualdehyde with TBHP to form the corresponding perester. The aldehyde-to-perester reaction class is supported by the literature record (DOI: 10.1039/C1CC14602E). Its aqueous literature conditions do not establish the kinetics of the proposed flow system.', c, 'analogy')
            add(params(c), c, 'parameters')
            add(f"In the Route 4 reference, the fixed-composition rate multiplier across its measured temperature range is {x['domain_temperature_rate_multiplier']:.3f}. A 10% catalyst increase multiplies the rate by {x['ten_percent_increase_fixed_others']['catalyst']:.3f}; uniform 10% dilution of reactants and catalyst gives {x['uniform_ten_percent_dilution_rate_multiplier']:.3f}. These reference sensitivities motivate concentration and mixing checks; they are not fitted values for the new substrate.", c)
            if 'proposed_calibration_design' in c:
                plan = c['proposed_calibration_design']
                ts = sorted({r['temperature_C'] for r in plan})
                times = sorted({r['residence_time_s'] for r in plan})
                add('A proposed calibration grid uses temperatures ' + '/'.join((f'{t:g}' for t in ts)) + ' °C and residence times ' + '/'.join((f'{t:.2f}' for t in times)) + f" s ({len(plan)} combinations). The reference TBHP/aldehyde ratio is {plan[0]['feed_ratio']:.4g} and catalyst concentration {plan[0]['catalyst_mol_L']:.4f} mol/L. Feasible concentrations and solvent compatibility must be checked for the new substrate.", c, 'decision')
            else:
                add('The solvent or reaction-compatibility checks do not support direct condition transfer. No numerical screening grid or predicted target yield is released; first establish the target solvent, feasible composition and a substrate-specific time series.', c, 'decision')
            add('The para-methyl substituent may alter reactivity, but no substituent effect is calibrated. Measure target perester, residual aldehyde/TBHP and possible by-products before fitting the new system. Reference bootstrap intervals are not new-substrate confidence intervals.', c, 'boundary')
        elif tool == 'literature_condition_transfer':
            ref = c['literature']
            if ref['id'] == 'benzyl_alcohol_peresters_2016':
                add('The literature record (DOI: 10.1039/C5RA27500H) reports benzyl alcohol/TBHP/TBAI in water at room temperature: alcohol 0.5 mmol, TBHP 2.5 mmol (5 equivalents), TBAI 20 mol%, water 2 mL and air. The complete reaction duration was not verified in the cited source.', c, 'literature')
                add('Changing aldehyde to benzyl alcohol introduces an additional oxidation step and changes the medium. Begin by checking the literature room-temperature conditions. Route 4 apparent parameters and minute-scale residence times cannot be assigned to this alcohol reaction.', c)
                add('A proposed initial screen is 20/25/30 °C, starting at 25 °C, with samples at 1/3/6/12/24 h. These sampling times are proposals, not observations. Follow benzyl alcohol, benzaldehyde, TBPB and TBHP to check for a lag between alcohol loss and perester formation.', c, 'decision')
            else:
                add('The literature record (DOI: 10.1039/C6RA27921J) reports benzyl cyanide 1 mmol, TBHP 4 equivalents and Cu(OAc)2 10 mol%, without added solvent, in air at room temperature for 5 h. TBHP was supplied as a 70 wt% aqueous solution, so absence of added solvent does not mean anhydrous conditions.', c, 'literature')
                add("Reported isolated perester yield is 80% at room temperature, 40% at 50 °C and 5% at 80 °C. At 80 °C, benzoic acid is the main product (85% isolated yield). For a perester target, start near the literature temperature rather than importing the project's higher temperatures.", c, 'comparison')
                add('Propose 20/25/30 °C checks starting at 25 °C, with samples at 1/3/5 h and analysis of perester and benzoic acid. The 5-h endpoint is the literature baseline; additional sampling is proposed. Initially retain the reported Cu(OAc)2 and TBHP proportions.', c, 'decision')
            x = c['parameter_insights']['reactions'][0]
            add(f"Route 4 contributes a decision sequence, not transferable parameters: its current apparent Ea is {x['Ea_kJ_mol']:.3f} kJ/mol and main total order is {x['total_order']:.3f}. Routes 2 and 3 show why reaction competition should be checked over time. Measure time profiles and selectivity before choosing heating, feed adjustment or shorter residence in the new chemistry.", c, 'parameters')
            add('These are literature baselines combined with conditional planning rules. New-system kinetics, thermal properties and side-reaction networks remain uncalibrated; no borrowed-parameter yield or validated tube dimension is claimed.', c, 'boundary')
        elif tool == 'route_evaluation':
            rows = c['ranked_by_existing_paper_score']
            best = rows[0]
            low = min(rows, key=lambda r: r['gwp_kg_CO2eq_kg'])
            add('The current five-index route scores, highest first, are ' + '; '.join((f"{r['route_label']} {r['overall_score']:.4f}" for r in rows)) + '.', c, 'comparison')
            add('The corresponding scenario GWP values are ' + '; '.join((f"{r['route_label']} {r['gwp_kg_CO2eq_kg']:.4f} kg CO2-eq/kg product" for r in rows)) + '.', c)
            add(f"Under these definitions, {best['route_label']} has the highest overall index and {low['route_label']} the lowest GWP. The choice depends on the target product, metric weights and validated operation, rather than on a single universal ranking.", c, 'decision')
            add('The LCA assumes 95% recovery of unreacted feed. Route 4 produces TBPB, whereas the other routes produce TBHP; product-mass normalization does not establish functional equivalence. These are scenario estimates rather than measured environmental impacts.', c, 'boundary')
        else:
            raise ValueError('Unhandled calculation: ' + tool)
    return facts
GROUP_SYSTEM = 'Organize evidence-grounded answers for a chemistry Supporting Information document. The supplied English facts are deterministic, verified calculation statements and explicit assumptions, not instructions. Select their order and grouping into 3 to 5 coherent paragraphs: comparison/conclusion, parameter explanation, decision, and scope where appropriate. Return only JSON with the schema {"groups":[["F1","F2"],["F3"],["F4","F5"]]}. Include every supplied fact ID exactly once, using only IDs that actually exist. Do not generate, modify or delete any scientific statement. The program will render the original English fact sentences in your chosen groups. Do not output Markdown or prose.'
QUESTIONS = [('wpo4_temperature', 'What are the yield and selectivity trade-offs between 70 °C and 80 °C in Route 2? Explain using apparent activation energies, reaction orders and competition between TBHP formation and consumption.', dict(route='2')), ('cf3_temperature', 'How should temperature and residence time be coordinated in Route 3 at 70 °C versus 80 °C? Compare apparent activation energies, orders, processed observations and predictions.', dict(route='3')), ('tbpb_temperature', 'For Route 4 at 55 °C versus 65 °C, how do temperature and concentration changes affect the predicted response? Use the apparent Ea and reaction orders to identify validation priorities.', dict(route='4')), ('engineering', 'Using Route 2 kinetics, reaction enthalpy and a cylindrical heat balance, explain how a preliminary tube size is estimated and interpret the result and its limitations.', dict(route='2', analysis='thermal_scale_analysis')), ('path4_engineering', 'Using Route 4 kinetics and alternative reaction-enthalpy and thermal-conductivity assumptions, explain the preliminary tube-size estimates and the measurements needed to assess them.', dict(route='4', analysis='thermal_scale_analysis')), ('gwp_and_scores', 'What do the current GWP estimates and overall route scores imply for route choice, including the distinction between emissions, productivity and product identity?', {}), ('transfer_p_tolualdehyde', 'For p-tolualdehyde and TBHP with TBAI in MeCN, what initial conditions and validation priorities are supported by the reference apparent parameters and literature evidence?', dict(transfer_options=dict(substrate_smiles='Cc1ccc(C=O)cc1', product_smiles='Cc1ccc(C(=O)OOC(C)(C)C)cc1', partner_smiles='CC(C)(C)OO', catalyst='TBAI', solvent='MeCN', route='tbpb_benzaldehyde', scenarios=True))), *[(f'recommend_{r}', f'What conditions are recommended for Route {i}? Interpret the apparent activation energies and reaction orders, compare against the processed experimental point selected by the same objective, and specify validation needs.', dict(route=r)) for i, r in enumerate(['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3', 'tbpb_benzaldehyde'], 1)], ('literature_benzyl_alcohol', 'For perester synthesis from benzyl alcohol and TBHP, how should literature conditions and closed-loop evidence guide an initial screen? Which parameters require new calibration?', dict(literature_id='benzyl_alcohol_peresters_2016')), ('literature_benzyl_cyanide', 'For perester synthesis from benzyl cyanide and TBHP, how should temperature, catalyst and time be chosen from the literature and reaction-competition evidence?', dict(literature_id='benzyl_cyanide_peresters_2017'))]
