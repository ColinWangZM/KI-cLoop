# Evidence-grounded literature extraction

The following input contract, prompts and validation rules provide the implementation protocol for the extraction workflow summarized in Supplementary Information section S1.4 and Supplementary Figs. 3–4. The tables below belong to this repository protocol; they are not numbered supplementary tables in the manuscript. These templates specify the intended workflow and do not establish which model generated a particular corpus record.

## Literature extraction and record structure

The literature-processing task begins with a domain definition and targeted retrieval terms. Retrieved passages are interpreted with the corresponding ontology to identify material entities, properties, synthesis steps and typed relations.

The protocol in this section specifies the extraction inputs, output schema, normalization and review rules. The prompt templates make the ontology-informed task explicit and provide a reusable implementation specification.

Supplementary Fig. 3 | Literature-to-record extraction protocol. The workflow comprises passage retrieval, ontology-guided extraction, field validation, evidence alignment, human review and graph record assembly.

Document preparation and evidence units

A document record contains its DOI or patent identifier, title, document version and retrieval metadata. Text paragraphs, table rows and figure captions are separate evidence units.

### Extraction input contract


| Input | Contents |
| --- | --- |
| DOCUMENT_METADATA | Document ID, DOI/patent number, title, version and page map |
| EVIDENCE_UNITS | Selected text/table units with stable span IDs |
| ONTOLOGY | Allowed entity types, relation types and property definitions |
| TARGET_SCOPE | Material class or reaction/property scope |
| OUTPUT_SCHEMA | Required keys, types, nullable fields and evidence pointers |


Document content is processed as evidence. Missing values are represented explicitly rather than inferred from conventional chemistry or from the language model’s background knowledge.

Extraction system prompt

The following template defines evidence-grounded extraction. For every entity, quantity and relation, include the supporting evidence-span ID. Preserve the source wording and the original numerical value and unit. Use null for information not stated in the evidence. Do not infer a reactant, catalyst, solvent, yield, condition or mechanism from general chemical knowledge. Preserve ranges, inequalities, uncertainty and negation. Keep observations from distinct experiments or sources separate.

### Prompt clauses and their purpose


| Clause | Purpose |
| --- | --- |
| Evidence-span ID per assertion | Allows a reviewer to locate the supporting passage |
| Preserve original value and unit | Prevents normalization from destroying the source record |
| Null when not stated | Distinguishes missing information from an estimated value |
| Preserve negation and qualification | Avoids turning a failed or conditional observation into a positive assertion |
| Separate experiments and sources | Prevents accidental mixing of conditions and outcomes |


Domain task prompt and review prompt

TARGET_SCOPE: {material_class_or_reaction_scope}DOCUMENT_METADATA: {document_metadata_json}ONTOLOGY: {entity_types_relation_types_property_definitions}EVIDENCE_UNITS: {span_id_and_source_text_or_table_cells}OUTPUT_SCHEMA: {schema_json}Identify each supported material and reaction event. Associate quantities with the correct entity, event and operating condition. For synthesis procedures, preserve the order of explicitly stated operations. For each quantity, return raw_text, value, unit, qualifier, context and evidence_span_ids. For each relation, return subject_id, predicate, object_id and evidence_span_ids. Return records, unresolved_mentions and review_flags. Do not complete missing fields with plausible values.

A separate review pass checks the extracted record against the same evidence. 

Structured extraction output

Supplementary Fig. 4 | Three-layer extraction record. Material identity, observation context and source evidence remain linked. Each row represents a different record layer, and arrows indicate links between layers.

Entity normalization and chemical identity

Normalization resolves alternative names while retaining the original mention. An alias is linked to a canonical identity only when the identifier, structure or unambiguous source context supports that link. Chemical form, charge, solvation and stereochemistry can change identity and are therefore preserved as qualifiers.

The numerical normalization layer records both raw and standardized quantities. Unit conversion is performed only when the source unit and quantity definition are resolved. Mass fraction, molar fraction, molarity and equivalents are different quantities and cannot be interchanged without the additional information needed for the conversion.

### Normalization rules


| Input feature | Representation |
| --- | --- |
| Abbreviation | Store as an alias of a supported identity |
| Salt, hydrate or solvate | Preserve the stated chemical form |
| Structure identifier | Retain source identifier and normalized identifier separately |
| Range or inequality | Keep bounds and comparator; do not replace by the midpoint |
| Yield | Preserve isolated/analytical basis and limiting reagent when stated |
| Catalyst loading | Keep mol%, mass%, equivalents or concentration as reported |
| Temperature | Store source scale and converted kelvin value |
| Reaction time | Associate with the stated operation or residence-time definition |


## Evidence validation

Evidence validation and manual review

Manual review covers representative records and cases flagged for identity or association ambiguity. 

### Review outcomes


| Outcome | Interpretation | Record treatment |
| --- | --- | --- |
| Supported | The evidence supports the assertion and context | Retain the assertion and evidence pointer |
| Partially supported | Some fields are supported; others are unstated | Retain supported fields and null the others |
| Ambiguous | Several assignments remain possible | Retain the source mention and review flag |
| Unsupported | The supplied evidence does not support the assertion | Exclude the unsupported assertion from the accepted record |


Repeated descriptions of one experiment are linked through their document and event identifiers. Alternative observations remain separate.

Worked example of evidence mapping

This example illustrates the output contract using the Route 4 description in the supplied manuscript. 

### Example mapping from the manuscript Route 4 description


| Record field | Mapped value | Evidence basis |
| --- | --- | --- |
| Reaction target | TBPB | Route 4 description |
| Reactants | Benzaldehyde; TBHP | Explicitly named reactants |
| Additive | TBAI | Explicitly stated system component |
| Solvent | Acetonitrile | Explicitly stated medium |
| Temperature, duration and yield | Null for this selected passage | Not supplied in the selected route-description sentence |


The passage identifies a chemical system but does not by itself specify an executable experimental recipe. Temperature, concentration and residence-time observations are attached from their own supporting experimental records. 

### Validation examples


| Proposed assertion | Decision |
| --- | --- |
| TBHP is a reactant in Route 4 | Supported by the selected passage |
| The selected sentence reports a 93% experimental yield | Unsupported by the selected passage |
| Acetonitrile is a product | Incorrect role assignment |
| The reaction has a specified duration | Unstated in the selected passage |

