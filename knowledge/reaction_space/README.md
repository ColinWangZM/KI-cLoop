# Reaction-space panel

Install `knowledge/reaction_space/requirements.txt`, then run from the repository root:

```bash
python knowledge/reaction_space/plot.py
```

The script exports annotated and unannotated reaction-space panels and legends to `knowledge/reaction_space/outputs/figures/`.

The panel contains 12,188 source records, including 26 peroxide-product records. Ten transformation classes and an unclassified/other group color a shared t-SNE projection. Five circled examples indicate low support in the original fingerprint space relative to 23,542 reference reactions.

Rendering uses saved coordinates. Projection settings, including perplexity 1,000 and seed 42, are recorded in `data/projection.json`.
