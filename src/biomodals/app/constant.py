"""Constants used across Biomodals apps."""

from modal import Volume

# Volume for caching all model weights
MODEL_VOLUME = Volume.from_name("biomodals-store", create_if_missing=True)

# Max timeout for any function, in seconds (24 hours)
MAX_TIMEOUT = 86400
