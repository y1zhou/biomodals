"""FlowPacker task implementation for the PPIFlow workflow."""

from biomodals.app.fold import flowpacker_app
from biomodals.schema import AppRunResult, ExecutionArtifact
from biomodals.workflow.ppiflow import staging
from biomodals.workflow.ppiflow.runtime_context import (
    SOURCE_VOLUME_ROOTS,
    reload_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    optional_config_int,
    patterns_from_config,
)


def run_ppiflow_flowpacker_stage(
    *,
    artifacts: list[ExecutionArtifact],
    config: dict[str, object],
    run_name: str,
) -> AppRunResult:
    """Select PPIFlow structures and invoke the FlowPacker app."""
    reload_source_volumes()
    selected = staging.select_structure_files_from_artifacts(
        artifacts,
        SOURCE_VOLUME_ROOTS,
        patterns=patterns_from_config(config),
        max_files=optional_config_int(config, "max_structures"),
    )
    kwargs = {
        key: config[key]
        for key in (
            "model_name",
            "use_confidence",
            "n_samples",
            "num_steps",
            "sample_coeff",
            "use_gt_masks",
            "inpaint",
            "save_traj",
            "seed",
        )
        if key in config
    }
    return AppRunResult.model_validate(
        flowpacker_app.run_flowpacker_workflow.get_raw_f()(
            input_files=selected,
            run_name=run_name,
            **kwargs,
        )
    )
