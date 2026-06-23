# Keep PPIFlow candidate ids out of DAG hashes

PPIFlow candidate ids are runtime provenance for produced artifacts, not semantic workflow DAG configuration. Changing candidate-id helper internals is a manifest migration concern unless user-facing workflow configuration changes; candidate ids and candidate manifests must not be added to node hash payloads.
