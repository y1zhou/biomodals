# App Testing and Verification

Test from deterministic policy outward. Source-string checks do not prove patch
application, resumption, or scientific results.

## Test pyramid

1. **Pure policy:** validation, safe IDs, identities, plans/manifests, budgets,
   parsers/reducers, ordering, and invariants.
2. **Artifact lifecycle:** partial/corrupt state, missing shards, stale workload
   publication claims, provider redelivery, Successor recovery, cleanup
   ordering, and force isolation.
3. **Composition:** discovery, config/image/mount wiring, primitive payloads,
   orchestration, and fake remotes/`get_raw_f()`.
4. **Upstream:** apply patches to the exact pinned commit, compile/import them,
   and compare raw plus final artifacts with an external oracle.
5. **Modal:** exercise the real image, mounts, warm containers, nested fanout,
   resume path, and downloaded output.

Cover multi-record and adversarial names/alphabets/paths/sizes; same-key callers;
force races; artifact/marker preemption; identity changes; cold/warm caches.

Prefer behavior over source searches; reserve source guards for textual
contracts such as a pinned patch preimage.

Keep tests aligned with the maintained contract. When a change removes a
field, flag, alias, compatibility path, or implementation, delete its tests in
the same change and cover the replacement behavior directly. Do not preserve
history with assertions that the retired symbol or value remains absent.
Reserve absence assertions for enduring boundaries such as secret redaction,
path containment, and an intentionally private public API surface.

## Output assertions

- Parse tables with Polars; assert schema/types/rows/nulls/unique IDs/joins/order
  and numeric tolerances. Validate archive members and containment.
- Assert each manifest's artifacts and cache/model/reference identities.
- Assert workflow schemas/storage/string paths and unchanged standalone CLI.
- Assert fanout task set/budget/retry and merge independence from finish order.

## Repository verification

Update `examples/app/` when invocation changes; reuse small fixtures and exclude
generated or large outputs.

Run the focused tests and static checks, then:

```bash
uv run biomodals app list
uv run biomodals app help <app-name>
uv run biomodals workflow list
prek run --files <changed-files>
```

Inspect production-shaped artifacts and scientific invariants. Report untested
paid GPU runs and full-reference benchmarks.
