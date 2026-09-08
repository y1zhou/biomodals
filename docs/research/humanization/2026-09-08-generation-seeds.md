# Humanization generation seeds and observed yield

Research date: 2026-09-08. Read-only local source, pinned upstream source, and
existing cached artifacts; no new scientific execution. This is evidence and a
proposal, not an accepted API change.

## What the controls actually do

| Method | One invocation per parent currently produces | Meaning of seed |
| --- | --- | --- |
| Humatch | One optimized paired endpoint, possibly unchanged | No seed or random sampling control |
| p-AbNatiV2 | One optimized paired endpoint, possibly unchanged | Root seed initializes seedable libraries through a derived pair seed; not a native candidate-count control |
| HuDiff | 1–25 paired sampling attempts in one tensor batch | Root seed initializes one random stream through a derived pair seed; attempts are batch rows, not separately seeded invocations |

The workflow schedules one generator task per method and parent and forwards
that method's settings unchanged. Current settings accept scalar unsigned
32-bit roots, default p-AbNatiV2 0 and HuDiff 42; HuDiff's candidate count
defaults to 10. A list cannot currently be submitted as Root seed.
Sources: [generation task discovery](../../../src/biomodals/workflow/humanization/workflow.py#L140),
[settings](../../../src/biomodals/workflow/humanization/settings.py#L38).

Both pair seeds hash the root, parent ID, VH, and VL. Therefore changing an ID
also changes the random stream, and scheduling order does not. HuDiff seeds
Python, NumPy, and Torch once, shuffles one shared mutable-position order when
requested, and draws residues with `torch.multinomial` across all batch rows.
It does **not** derive a separate seed per attempt. Changing the batch count
changes random-number consumption; do not promise that a larger batch retains
the exact smaller-batch prefix. Increasing attempts already creates additional
stochastic samples without entering multiple roots.
Sources: [HuDiff pair seed](../../../src/biomodals/app/design/hudiff_ab/worker.py#L66),
[batch sampling](../../../src/biomodals/app/design/hudiff_ab/upstream_runtime.py#L105),
[p-AbNatiV2 pair seed](../../../src/biomodals/app/design/pabnativ2/app.py#L339).

HuDiff validates every attempt, retains invalid/duplicate attempt evidence, and
returns only valid distinct paired sequences. It does not resample failures to
fill the requested count. Its standalone aggregate limit is 10,000 attempts;
the website's present 100-parent limit and 25-attempt limit imply at most 2,500
HuDiff attempts in one current workflow job. Any seed-list extension needs an
explicit aggregate budget rather than accidentally multiplying this ceiling.
Sources: [normalization](../../../src/biomodals/app/design/hudiff_ab/worker.py#L255),
[limits](../../../src/biomodals/app/design/hudiff_ab/app.py#L169).

p-AbNatiV2 calls its paired humanizer once, takes the final endpoint, and also
predicts/scores input and final structures. Its pinned optimizer evaluates
allowed substitutions and chooses by score and pairing constraints; the
comment mentioning Metropolis does not introduce a random acceptance draw.
Its public paired optimizer has no seed argument. Wrapper seeding is useful
for reproducibility of seedable runtime components, but is **not evidence that
different seeds will produce different sequences**. Repeated roots should be
described as optimization replicates, not guaranteed novel candidates. The
structure pipeline and tie behavior prevent a blanket cross-device
determinism guarantee.
Sources: [wrapper endpoint](../../../src/biomodals/app/design/pabnativ2/app.py#L532),
[pinned paired optimizer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py#L939),
[pinned substitution acceptance](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py#L1195).

Humatch first satisfies germline-likeness targets, then performs greedy
single-substitution search only while classifier targets are unmet. If the
parent already satisfies all targets, an unchanged endpoint is expected. If
search fails, upstream returns its best visited endpoint, which may also be
unchanged. `humanization_success` in the native summary means classifier
targets were satisfied, not that mutations occurred. There is no scientifically
meaningful seed-list knob for this greedy search; a different family or target
threshold changes the scientific objective instead.
Sources: [wrapper and summary](../../../src/biomodals/app/design/humatch/app.py#L571),
[pinned Humatch search](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py).

## Latest cached job: Humatch was a successful no-op

Examined existing job `a9085e43-5ad9-4234-a933-863e801ab869`, cached archive
`.biomodals/cache/results/a9085e43-5ad9-4234-a933-863e801ab869.result`.
All manifest sizes and SHA-256 digests were verified. Reconstructing the exact
paired-sequence union from native outputs reproduced all 10 selection rows;
no valid native output was removed by ranking.

| Source | Observed outcome |
| --- | --- |
| Parent | One baseline row, generation method null |
| Humatch | Successful endpoint identical to parent; zero edits; merges into baseline |
| Sapiens | One distinct changed pair |
| p-AbNatiV2 | One distinct changed pair, 11 mutations |
| HuDiff | 10 attempts: 7 valid unique pairs, 3 invalid, zero duplicates |

Humatch selected hv1/kv3. Parent VH, VL, and pairing probabilities were
0.9998815059661865, 0.9989855885505676, and 0.9899839162826538, all above 0.95.
Germline-likeness scores were 0.4123623261680902 and 0.41907539953814704, both
above 0.4. Native summary has `humanization_success=true`, `edit_count=0`;
input and humanized CSV bytes are identical, SHA-256
`724dcbe7c322ee3556d46343e08dc639220c2af6abd0117490e8a7e8a4fe5ac6`.
Evidence member:
`generation-db85a2bc7790a4e416131b64b3b57360c0cc7fb1be53bfa9653cbf421b0c59e3.zst`.

HuDiff root 42 yielded pair seed 4165055162. All three rejected attempts changed
IMGT grid occupancy; rejection was not ranking or method-balancing. Evidence:
`generation-82e250083323c496c3f24594332ac18437a0e1d35dde4836104cf4a90a10d996.json`.
p-AbNatiV2 root 0 yielded pair seed 3627496826. Evidence:
`generation-bf1b094276c7c8a1d87a57ab4136ab40ffb129b618563a8da7857d80e2876db9.json`.
All selection error columns were null.

The union deduplicates exact `(parent_id, vh, vl)` tuples across methods and
retains all origins internally. The display deliberately sets the parent's
generation method to null, even when a generator returned it. Thus absence of
a Humatch label is expected here, not missing execution or lost evidence.
Sources: [union](../../../src/biomodals/workflow/humanization/tables.py#L52),
[parent display](../../../src/biomodals/workflow/humanization/tables.py#L155),
[native decoding](../../../src/biomodals/workflow/humanization/artifacts.py#L57).

## Smallest truthful options for agreement

1. Keep HuDiff's existing root and attempt count when the goal is merely more
   samples. Relabel/help-text can clarify that attempts already sample a batch.
   More HuDiff attempts will not solve method imbalance or Humatch no-ops.
2. If users need explicit replicate roots, add bounded, unique root lists only
   for the two seeded methods, with one root preserving current behavior. Run
   p-AbNatiV2 once per root and label these optimization replicates; novelty is
   unknown and cannot be promised. Keep Humatch unchanged.
3. Before implementation, decide whether HuDiff attempts are **total per
   parent** split across roots, or **per root**. The latter is easier to compose
   from unchanged native calls but multiplies attempted yield and cost. Never
   silently reinterpret the existing count. A fixed total avoids accidental
   budget expansion but needs an explicit allocation rule and changes batch
   shapes (therefore potentially all sampled sequences).

For `P` parents, `R_p` p-AbNatiV2 roots, `R_h` HuDiff roots and `A` HuDiff
attempts per root, nominal generation calls become
`P * (2 + R_p + R_h)` instead of `4P`; HuDiff attempts become `P * R_h * A`.
The pre-dedup endpoint ceiling including baselines is
`P * (3 + R_p + R_h * A)`. Invalid samples, identical endpoints, and
cross-method overlap reduce actual yield. Three evaluators score the unique
union, so duplicate generation still costs work but need not duplicate
evaluation. More roots increase full optimizer/model invocations and native
evidence volume; HuDiff's current batched attempts can amortize work. These are
work-count formulas, not dollar estimates or guarantees of linear runtime.
Sources: [workflow tasks](../../../src/biomodals/workflow/humanization/workflow.py#L140),
[HuDiff batched inference](../../../src/biomodals/app/design/hudiff_ab/upstream_runtime.py#L126),
[union](../../../src/biomodals/workflow/humanization/tables.py#L52).

If adopted, use existing execution tasks and exact union deduplication, not a
new scheduler. Keep root and derived seed plus attempt index recoverable in
native evidence. Do not represent independently seeded invocations as HuDiff
batch attempts, force extra Humatch mutations merely to obtain a label, or
duplicate parent IDs to simulate seeds: IDs also scope deduplication/ranking.
No implementation is authorized by this note.
