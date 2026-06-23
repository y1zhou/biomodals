# Generate PPIFlow candidate ids from provenance

PPIFlow candidate ids are deterministic and provenance-based. Initial candidates hash the producing stage, source artifact id or path, and normalized file basename; derived candidates hash the parent candidate id, stage name, operation mode, and derived output basename. Sequential ids are reserved for synthetic stage-2 convenience manifests and must keep source-path provenance so users can reconcile them later.
