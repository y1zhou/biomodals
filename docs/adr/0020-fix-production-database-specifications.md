# Fix production database specifications

Status: accepted.

The production shard builder accepts one supported `database_id`,
`seqkit_threads` defaulting to 8, and `source_policy` defaulting to `keep`.
`database_id` selects a code-owned Supported Database Specification for one of
small BFD, MGnify, UniProt, UniRef90, NT-RNA, RFam, or RNAcentral. Each
specification fixes the official source filename, molecule type, accepted shard
count, and expected source statistics.

One Modal invocation builds and validates one logical database. Production does
not accept free-form source paths, shard counts, Z values, or polymer types;
changing any such scientific input requires a reviewed specification/profile
change and produces a new manifest identity. Experimental overrides remain in
the temporary `alphafold3_msa_app.py` harness until separately promoted.

This boundary prevents a mistyped runtime argument from publishing a
scientifically mislabeled profile while preserving the two operational controls
needed for setup cost and source retention.
