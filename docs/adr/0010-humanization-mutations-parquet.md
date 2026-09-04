# Standardize humanization mutation tables on Parquet

Status: accepted.

All humanization apps emit their parental-to-result substitution table as a
Parquet artifact. Sapiens names its iterative table
`mutation_history.parquet`; Humatch, p-AbNatiV2, and HuDiff-Ab use
`mutations.parquet`. Compact sequence, summary, and candidate-attempt tables
remain CSV so they are easy to inspect directly.

Mutation tables can grow with input pairs, generated candidates, and method
iterations. Parquet avoids scaling their wire size as wide text while retaining
typed columns and efficient downstream workflow reads. The common storage
format does not erase method-specific semantics: each app keeps the coordinates,
endpoint or iteration fields, and scientific identity required to interpret
its own mutations. Existing humanization bundle manifests advance to schema
version 2 with this artifact change.
