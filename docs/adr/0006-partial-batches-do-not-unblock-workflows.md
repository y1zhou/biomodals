# Preserve partial batch results without treating them as success

Batch app adapters will return succeeded only when every requested candidate succeeds, partial when successful and failed candidates are mixed, and failed when none succeed. Partial results retain successful outputs, failure records, and logs, but remain terminal non-success so downstream scientific steps never consume an incomplete candidate or score set implicitly.
