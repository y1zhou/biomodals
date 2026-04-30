# PPIFlow Workflow Notes

`src/biomodals/workflow/ppiflow_workflow.py` mirrors the upstream PPIFlow scheduler from <https://github.com/Mingchenchen/PPIFlow> while delegating heavy steps to deployed Biomodals apps.

## App Dependencies

The workflow looks up deployed functions by app name. Defaults are:

- `PPIFlow` for backbone generation and partial flow refinement.
- `LigandMPNN` for ProteinMPNN and AbMPNN-compatible sequence design.
- `FlowPacker` for side-chain packing.
- `AF3Score` for score-only AlphaFold3 confidence metrics.
- `AlphaFold3` for ReFold.
- `Rosetta` for Rosetta relax jobs.
- `DockQ` for model/reference DockQ scoring.

Each app name can be overridden with an environment variable such as `PPIFLOW_APP_NAME` or `DOCKQ_APP_NAME`.

The workflow currently depends on these workflow-oriented deployed functions in addition to existing app entrypoints:

- `PPIFlow.ppiflow_run_bytes(...)` runs PPIFlow from in-memory input files and returns a packaged `.tar.zst` archive.
- `AF3Score.af3score_run_bytes(...)` scores in-memory PDB files and returns AF3Score outputs plus `af3score_metrics.csv`.
- `Rosetta.run_rosetta_batch_bytes(...)` runs a batch of in-memory Rosetta jobs and returns packaged outputs.
- `DockQ.run_dockq_batch(...)` scores in-memory model/reference pairs and returns `dockq_results.csv`.

## Workflow-Specific Tradeoffs

- The public interface uses upstream `task.yaml` and `steps.yaml` files. This keeps PPIFlow examples portable and avoids a second workflow schema.
- Stage directories preserve upstream names so downstream scripts can inspect `stage1/`, `stage2/`, and `design_output/`.
- LigandMPNN is used as the Biomodals ProteinMPNN/AbMPNN executor. Binder workflows map to `protein_mpnn` or `soluble_mpnn`; antibody and nanobody workflows map to `abmpnn`.
- The current FlowPacker app accepts structure files rather than the upstream PPIFlow `mpnn_seqs.csv` handoff. The workflow asks LigandMPNN to pack side chains when possible and falls back to the current backbone PDBs if no packed PDBs are emitted before FlowPacker.
- Rosetta fix and relax reuse the generic Rosetta app. Because the generic wrapper does not produce upstream `residue_energy.csv`, `fixed_positions.csv` is populated from `PartialStep.fixed_positions` or `task.fixed_positions` when provided; otherwise each row is set to `NONE` and partial flow copies the input structure unchanged.
- ReFold is implemented with the Biomodals AlphaFold3 app by extracting protein chain sequences from filtered PDBs and generating AlphaFold3 JSON inputs. The workflow always goes through the AlphaFold3 data-pipeline helper before inference.
- DockQ pairs refolded model structures with filtered references by matching reference stems in the refold output path. Runs with ambiguous model/reference names may need manual inspection of `stage2/dockq_output`.
