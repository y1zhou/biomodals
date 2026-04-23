"""Constants used across Biomodals apps."""

from modal import Volume

# Volume for caching all model weights
MODEL_VOLUME = Volume.from_name("biomodals-store", create_if_missing=True)

# Volume for caching MSA databases, which are large and shared across apps
AF3_MSA_DB_VOLUME = Volume.from_name(
    "AlphaFold3-msa-db", create_if_missing=True, version=2
)
PROTENIX_MSA_DB_VOLUME = Volume.from_name(
    "Protenix-msa-db", create_if_missing=True, version=2
)

# Volume for caching MSA search results
MSA_CACHE_VOLUME = Volume.from_name(
    "biomodals-msa-cache", create_if_missing=True, version=2
)

# Max timeout for any function, in seconds (24 hours)
MAX_TIMEOUT = 86400
