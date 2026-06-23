# Store PPIFlow candidate manifests as workflow artifacts

PPIFlow candidate manifests are first-class workflow artifacts stored in the workflow run volume, not only node metadata. Metadata can summarize manifest paths and counts, but downstream nodes and users need a durable table artifact for candidate joins, retry skipping, provenance inspection, and stage-2-only inputs.
