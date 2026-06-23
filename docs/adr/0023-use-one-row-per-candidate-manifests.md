# Use one-row-per-candidate PPIFlow manifests

PPIFlow candidate manifests store one Parquet row per candidate with a nested list of file records. Candidate-level joins, filtering, ranking, and reporting are the common operations, and file-level availability checks can expand the nested file list when needed.
