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

## Workflow-Specific Tradeoffs

- The public interface uses upstream `task.yaml` and `steps.yaml` files. This keeps PPIFlow examples portable and avoids a second workflow schema.
- Stage directories preserve upstream names so downstream scripts can inspect `stage1/`, `stage2/`, and `design_output/`.
- LigandMPNN is used as the Biomodals ProteinMPNN/AbMPNN executor. Binder workflows map to `protein_mpnn` or `soluble_mpnn`; antibody and nanobody workflows map to `abmpnn`.
- ReFold is implemented with the Biomodals AlphaFold3 app by extracting protein chain sequences from filtered PDBs and generating AlphaFold3 JSON inputs. Template stripping is represented by creating JSON inputs without templates.
- DockQ is provided as a Biomodals app wrapper so the workflow can keep external compute steps behind deployed functions.
- Rosetta fix and relax reuse the existing Rosetta app where possible. Any additional workflow helper must preserve the upstream output contracts needed by fixed-position generation and ranking.
