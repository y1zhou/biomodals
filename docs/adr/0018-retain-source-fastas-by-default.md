# Retain source FASTAs by default

Status: accepted.

The production shard builder exposes a `source_policy` with `keep`, `compress`,
and `delete` values. `keep` is the default. The original source FASTA is never
changed until its Sharded Database Profile has been published, committed, and
deeply revalidated. The temporary source copy needed for two-pass shuffling is
builder staging and is excluded from the published profile.

`compress` appends `.zst` to the complete source filename, writes the archive
beside the original, and verifies that decompression reproduces the recorded
source byte count and SHA-256 before committing the archive and deleting the
plain FASTA. A compression or verification failure leaves the original intact
and does not advertise the archive as complete.

The builder only accepts the official uncompressed source FASTA. If that file
is absent but its `.zst` archive exists, it raises an actionable error asking
the user to restore the archive manually, for example in a Modal Sandbox,
before retrying. The app does not provide a restore function and never performs
an implicit full-database decompression or hidden Volume write.

`delete` removes the original only after the same profile checks pass and only
when explicitly selected. Both destructive policies record their outcome
durably. Source retirement does not change the profile identity because the
manifest retains the original source digest, size, sequence statistics, and
construction recipe.
