# Keep MSA cache orchestration in Biomodals

Status: accepted.

Biomodals owns Raw Database MSA cache discovery, missing-search scheduling,
validation, and publication. The pinned AlphaFold source remains unaware of
Modal Volume paths and cache markers; in particular,
`_get_protein_msa_and_templates` will not be patched to discover the cache.
That private function couples four database searches, MSA assembly, and
template search behind a process-local cache, so placing durable orchestration
inside it would cross the application boundary and defeat one-worker-per-
database scheduling.

An app-owned, version-coupled adapter reuses the pinned upstream assembly
semantics. It deduplicates protein unpaired results in UniRef90, small BFD, and
MGnify order; preserves the UniProt paired result without deduplication; and
deduplicates RNA unpaired results in RFam, RNAcentral, and NT-RNA order.
Protein template search runs from the completed combined unpaired MSA. If a
source patch is needed, it may expose a narrow pure helper seam but must not
introduce Modal storage knowledge into AlphaFold.

This choice makes upstream-version coupling explicit. Contract tests must bind
the adapter to the pinned source and compare its combined outputs with that
source's own pipeline before an AlphaFold pin change is accepted.
