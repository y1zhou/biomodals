# Keep caller search evidence request-local

Status: accepted.

Caller-Supplied Search Evidence is materialized only into its request's
Enriched AlphaFold Input and AlphaFold Run Root; it is never published into the
shared sequence cache. Database searches performed to fill other missing
fields still publish their independently valid Raw Database MSAs for reuse.

A Combined MSA Publication may replace the sequence-root files only when every
constituent alignment came from validated canonical raw results. A Template
Search Result may replace the sequence-root template publication only when its
input was the canonical Combined Unpaired MSA. Mixed caller/generated
assemblies and templates remain request-local.

This boundary avoids letting custom or path-supplied biological evidence
silently replace the app's shared default results while retaining useful
per-database work after partially generated requests.
