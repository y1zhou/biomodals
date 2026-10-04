# Model acceptance artifacts

These commands run the current source on Modal and require configured credentials
and model assets. They are manual GPU checks, not part of ordinary CI.

## FlowPacker

```bash
OUTPUT_DIR=/absolute/path/flowpacker-evidence bash examples/app/flowpacker.sh
```

The retained archive contains structures, upstream metrics, configuration, logs,
and `validation.json` with input/checkpoint digests and requested sample coverage.
The separate acceptance report reopens the archive, parses every declared PDB,
and checks coverage and content digests. Failure logs remain on the output volume.
This validates publication integrity; it does not measure packing accuracy.

## OligoFormer

```bash
OUTPUT_DIR=/absolute/path/oligoformer-evidence bash examples/app/oligoformer.sh
```

The archive includes final tables and `provenance.json`: upstream commit,
evaluation policy, model identity, efficacy checkpoint digest, input digests,
settings, efficacy cache key, and table digests. The report checks finite scores,
unique candidate positions, descending ranking, and the filtered candidate set.
A single run reports `scientific_comparison: not_run`.

For the numerical acceptance gate, independently clone upstream commit
`e2f53ad63387bbe166bf123949151e2bc9bf6ec3`, use the same RNA-FM and efficacy
weights and input FASTA, call `best_model.eval()` after weight loading, and run
`infer` under `torch.inference_mode()`. Run its native `scripts/main.py` inference
with seed 42 and toxicity enabled. Keep the generated `.txt` tables in a separate
oracle directory, along with the command, environment, input and weight digests.
Do not create this oracle using a Biomodals run. Check those input/weight digests
against the retained Biomodals provenance before accepting the comparison.

```bash
ACCEPTANCE=1 ORACLE_DIR=/absolute/path/independent-oracle \
  OUTPUT_DIR=/absolute/path/oligoformer-evidence \
  bash examples/app/oligoformer.sh
```

This runs two fresh forced generations, rejects identical efficacy cache keys,
checks matching input/model/settings provenance, and compares all repeated tables.
It compares the independent oracle's candidate identities and efficacy scores;
it does not claim equivalence for Biomodals' separately versioned filter policy.
The default absolute score tolerance is `1e-5`, with zero relative tolerance.
Candidate/ranking order must agree between repeated runs. Hardware-specific
tolerances must be justified explicitly; the validator never widens them itself.
The report retains archive and oracle-table digests and maximum score differences.

## Runtime compatibility without GPUs

CI uses `runtime_requirements.py` to export requirements through the same helper
as image builds, and `stage_rna_runtime.py` to materialize declared source mounts.
`rna_runtime.py` exercises request serialization, execution-plan construction,
and worker-plan transport under Python 3.10 and 3.11 while app entrypoint imports
are blocked. CI uploads the request bytes and verification reports.
