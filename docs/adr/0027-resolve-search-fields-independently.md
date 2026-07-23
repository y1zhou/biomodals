# Resolve search fields independently

Status: accepted.

When MSA search is enabled, the coordinator resolves every empty search field
without replacing a non-empty sibling. A missing protein unpaired MSA requires
only the UniRef90, small-BFD, and MGnify searches; a missing protein paired MSA
requires only the UniProt search. A non-empty RNA unpaired MSA suppresses its
RFam, RNAcentral, and NT-RNA searches.

Requested protein-template search runs after unpaired-MSA resolution and uses
the resolved unpaired alignment. If that alignment is Caller-Supplied Search
Evidence, the resulting template list remains request-local under ADR 0026.
The coordinator does not run searches for already populated fields merely to
create a canonical shared publication.

Independent resolution preserves the caller's biological evidence and avoids
unnecessary database work while retaining canonical Raw Database MSAs from
every genuinely missing field. Search-disabled empty-field behavior remains
defined by ADR 0010.
