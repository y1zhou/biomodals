"""Shared constants used across Biomodals apps and workflows."""

from modal import Volume

# Volume for caching all model weights.
MODEL_VOLUME_NAME = "biomodals-store"
MODEL_VOLUME = Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# Volume for caching MSA databases, which are large and shared across apps.
AF3_MSA_DB_VOLUME = Volume.from_name(
    "AlphaFold3-msa-db", create_if_missing=True, version=2
)
PROTENIX_MSA_DB_VOLUME = Volume.from_name(
    "Protenix-msa-db", create_if_missing=True, version=2
)

# Volume for caching MSA search results.
MSA_CACHE_VOLUME_NAME = "biomodals-msa-cache"
MSA_CACHE_VOLUME = Volume.from_name(
    MSA_CACHE_VOLUME_NAME, create_if_missing=True, version=2
)

# Durable workflow-orchestrator output ledger/artifact volume.
WORKFLOW_ORCHESTRATOR_VOLUME_NAME = "biomodals-workflow-orchestrator"
WORKFLOW_ORCHESTRATOR_VOLUME = Volume.from_name(
    WORKFLOW_ORCHESTRATOR_VOLUME_NAME, create_if_missing=True, version=2
)

# Max timeout for any function, in seconds (24 hours).
MAX_TIMEOUT = 86_400
