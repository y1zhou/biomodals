"""Mounted Volume context shared by PPIFlow task implementations."""

import modal

from biomodals.helper.constant import WORKFLOW_ORCHESTRATOR_VOLUME_NAME

PPI_FLOW_OUTPUT_VOLUME_NAME = "PPIFlow-outputs"
PPI_FLOW_OUTPUT_MOUNTPOINT = "/mnt/PPIFlow-outputs"
FLOWPACKER_OUTPUT_VOLUME_NAME = "FlowPacker-outputs"
FLOWPACKER_OUTPUT_MOUNTPOINT = "/mnt/FlowPacker-outputs"
AF3SCORE_OUTPUT_VOLUME_NAME = "AF3Score-outputs"
AF3SCORE_OUTPUT_MOUNTPOINT = "/mnt/AF3Score-outputs"
ALPHAFOLD3_OUTPUT_VOLUME_NAME = "AlphaFold3-outputs"
ALPHAFOLD3_OUTPUT_MOUNTPOINT = "/mnt/AlphaFold3-outputs"
ROSETTA_OUTPUT_VOLUME_NAME = "Rosetta-outputs"
ROSETTA_OUTPUT_MOUNTPOINT = "/mnt/Rosetta-outputs"
WORKFLOW_OUTPUT_VOLUME_NAME = WORKFLOW_ORCHESTRATOR_VOLUME_NAME
WORKFLOW_OUTPUT_MOUNTPOINT = "/mnt/WorkflowOrchestrator-outputs"

SOURCE_VOLUME_ROOTS = {
    PPI_FLOW_OUTPUT_VOLUME_NAME: PPI_FLOW_OUTPUT_MOUNTPOINT,
    FLOWPACKER_OUTPUT_VOLUME_NAME: FLOWPACKER_OUTPUT_MOUNTPOINT,
    AF3SCORE_OUTPUT_VOLUME_NAME: AF3SCORE_OUTPUT_MOUNTPOINT,
    ALPHAFOLD3_OUTPUT_VOLUME_NAME: ALPHAFOLD3_OUTPUT_MOUNTPOINT,
    ROSETTA_OUTPUT_VOLUME_NAME: ROSETTA_OUTPUT_MOUNTPOINT,
    WORKFLOW_OUTPUT_VOLUME_NAME: WORKFLOW_OUTPUT_MOUNTPOINT,
}
SOURCE_VOLUMES = {
    volume_name: modal.Volume.from_name(
        volume_name,
        create_if_missing=True,
        version=2,
    )
    for volume_name in SOURCE_VOLUME_ROOTS
}
WORKFLOW_OUTPUT_VOLUME = SOURCE_VOLUMES[WORKFLOW_OUTPUT_VOLUME_NAME]


def reload_source_volumes() -> None:
    """Reload every mounted Volume that may contain upstream publications."""
    for volume in SOURCE_VOLUMES.values():
        volume.reload()
