# Derive PPIFlow sequence tables in the workflow

PPIFlow derives `mpnn_seqs.csv` from LigandMPNN artifacts inside the workflow instead of adding PPIFlow-specific sequence-table semantics to the LigandMPNN app. The table is candidate-keyed and records parent provenance so stage 1 and stage 2 sequence outputs can be joined with structures, scores, ranking, and reports.
