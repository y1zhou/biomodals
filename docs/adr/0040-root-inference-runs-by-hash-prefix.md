# Root inference runs by hash prefix

Status: accepted.

An AlphaFold Run Root lives directly at
`/{run_id[:2]}/{run_id}/` in the app-specific AlphaFold3 output Volume. It does
not add a top-level `runs/` directory.

Each run root uses the following stable subtrees:

- `inputs/identity.json` for the seed- and display-name-neutral identity view;
- `custom-templates/{sha256}.cif` for Staged Custom Templates;
- `outputs/` for canonical Seed Predictions and global summary files;
- `outputs/.workers/{claim-generation}/` for Inference Worker Staging;
- `requests/{request_id}/` for request input, manifest, ranking, best files, and
  partial-failure evidence;
- `.markers/seeds/{seed}.json` and `.markers/summary.json` for completion;
- `logs/` and `metrics/` for durable operational evidence.

The two-character prefix only fans out Volume directories; the full `run_id`
remains the identity. Omitting `runs/` keeps the app-owned Volume hierarchy
consistent with the existing sequence-hash cache layout.
