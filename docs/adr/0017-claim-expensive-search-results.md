# Claim expensive search results atomically

Status: accepted.

Every missing expensive search-cache publication uses the same append-only
claim-generation protocol as Seed Predictions. A Raw Database MSA claim is
keyed by `(sequence_hash, database_id, search_identity)`. A Template Search
Result claim is keyed by `(sequence_hash, template_identity)`, where the
template identity already binds the completed unpaired MSA and template
reference snapshot.

The request that atomically creates the current Search Build Claim writes into
generation-exclusive staging and may publish only while it retains ownership.
Other requests wait for, reload, and validate the corresponding output-Volume
completion marker. Failed or conservatively expired work advances to another
generation without treating a claim record as scientific completion or
blindly deleting a possibly replaced owner.

This prevents independent submissions, duplicate homomer chains, or concurrent
coordinators from paying for the same HMMER or template search while preserving
the agreed one-database-result retry boundary.
