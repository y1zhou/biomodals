# Claim database profile builds minimally

Status: accepted.

The production shard builder first reuses an already published, matching
profile. Otherwise it elects one builder per Profile ID with an atomic Modal
Dict `put(..., skip_if_exists=True)`.

The claim uses small append-only owner and terminal-status records by
generation. It has no heartbeat or polling loop:

- an active-generation conflict fails immediately and reports the existing
  claim;
- a normal failure records `failed`, allowing the next explicit invocation to
  claim the next generation;
- an owner older than the builder's maximum possible function lifetime plus a
  conservative margin may be marked `abandoned`;
- concurrent takeover attempts still elect only one next-generation owner.

The builder never deletes an owner record, and only the validated
`manifest.json` proves publication. Different Profile IDs use independent
claims and may build concurrently. This narrow guard prevents an accidental
duplicate full-database build without introducing a general lease service for
an operation expected to run rarely.
