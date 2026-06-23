# Use Parquet for PPIFlow candidate manifests

PPIFlow candidate manifests are stored as Parquet table artifacts. They are internal provenance and join artifacts, so compact storage, fast Polars reads, typed columns, and nested field support matter more than line-oriented text inspection; upstream-facing tables such as `mpnn_seqs.csv`, AF3Score metrics, DockQ scores, and reports keep their existing formats unless separately changed.
