# Hand enriched JSON directly to inference

Status: accepted.

The Biomodals MSA coordinator becomes the complete CPU data stage. After
assembling all required protein MSAs, RNA MSAs, and protein templates, it
constructs and validates an Enriched AlphaFold Input and sends that JSON to the
existing inference function with `--run_data_pipeline=false`. AlphaFold still
performs featurization inside its inference path.

The production app therefore removes the old `run_data_pipeline` subprocess
path and its `copy_msa_to_ssd` behavior. It must fail closed when requested
search evidence is incomplete rather than allowing inference preparation to
silently replace missing searched fields with empty values. Explicit
single-sequence or no-template inference remains available through the search
policy recorded separately.
