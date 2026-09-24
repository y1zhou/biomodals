# AlphaFold3 chemistry inputs

Status: implementation approved 2026-09-24; verification pending.

Researched 2026-09-23 against backend `aa7d853` and the pinned AF3 source below.
This extends the input scope of [AlphaFold3 UX](alphafold3-ux.md), not its
prediction selection or viewer contract.

## Requested outcome

Add UI support for post-translational modifications and ligand–polymer covalent
bonds, including N-linked glycans and their internal sugar linkages. Explicit
disulfide constraints are deferred because the upstream model does not support
them. No inference, deployment or runtime
changes have been performed during research.

## Verified input and scientific boundaries

- Protein modifications replace a residue with a CCD component via
  `modifications: [{ptmType, ptmPosition}]`; positions are one-based within the
  submitted chain, not antibody numbering or PDB author numbering.
- Glycans require CCD ligand components and explicit attachment/internal bonds;
  a generic glycosylation flag does not specify their chemistry. Bond endpoints
  identify a chain, one-based component/residue index, and CCD atom name.
- SMILES ligands cannot be explicit bond endpoints because they lack stable
  input atom names. Custom named components can use inline `userCCD`.
- Explicit polymer–polymer bonds are unsupported. Crucially, the pinned AF3
  featurization removes these bonds, so a schema-valid cysteine SG–SG pair is
  not an enforced disulfide constraint. This is not fixed by adding UI fields.

Sources: [official input format](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#bonds),
[uniaf3 input format](https://github.com/y1zhou/uniaf3/blob/master/docs/alphafold3-input-format.md),
and [pinned featurization](https://github.com/y1zhou/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/pipeline/pipeline.py#L142).
The requested uniaf3 page was also inspected through its local source checkout;
the web fetch was unavailable. The model pin is defined in
`src/biomodals/app/fold/alphafold3/profiles.py`.

The pinned [RNase B example](https://github.com/y1zhou/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/examples/rnaseb_glycosylated.json)
provides an explicit `NAG,NAG,BMA,MAN,MAN` branched glycan attached through
ASN ND2 to the first NAG C1. This is a useful fixture or possible preset, not
evidence that the same glycoform or site occupancy applies to every protein.

## Existing seams and risks

- `service/alphafold3/validation.py` parses through `uniaf3` and retains the
  normalized document. Its preview already counts modifications and bonds.
  Structural parsing does not establish chemical atom validity or whether
  the model uses a particular bond.
- Frontend `src/alphafold3.ts` preserves Expert JSON fields except the documented
  visible name/seed overrides. Regular mode currently has no modification or
  bond editor. Preserve the existing separation of modes.
- Regular chain letters are derived from entity order/copy counts. New bond
  references must not silently change targets when entities are reordered or
  removed. Stable entity/copy references can be resolved at serialization.
- Both modes need truthful unsupported-bond feedback; a new regular editor
  alone would leave Expert input susceptible to the same silent loss.
- The installed uniaf3 0.2.1 validates endpoint IDs/ranges and SMILES exclusion,
  but not arbitrary CCD atom existence or polymer–polymer support. Disabling
  the upstream cleaning flag alone would not establish disulfide conditioning.
- GROMACS preparation currently removes non-protein atoms. This UI feature
  does not imply that glycosylated MD simulation is supported.

## Accepted direction (2026-09-23)

Use guided glycan presets at explicitly selected Asn positions, alongside
advanced protein PTM and supported ligand-bond rows. Show residue context and
one-based input positions; keep arbitrary custom chemistry in Expert JSON
initially.

Initial glycan presets are single NAG and the branched NAG2Man3 core shown in
the pinned RNase B example. No automatic site occupancy is inferred. In Regular
mode, PTMs and glycan attachments apply to every copy of the selected entity;
asymmetric modifications require separate entities. Disulfide enforcement is
explicitly deferred; this release does not patch the upstream model.
Use model-native CPU preflight to check manually entered CCD components and
atom names before expensive prediction work. Run it when the user clicks
**Continue**, before the existing confirmation page, with explicit
**Checking modifications and covalent bonds…** progress. Do not add another
confirmation screen. No remote execution is authorized merely by this design
decision.

## Approved implementation plan

1. Extend the Regular draft with protein PTM rows, the two approved glycan
   presets, and advanced supported bond endpoints. Offer common verified PTM
   choices (SEP, TPO, PTR, ALY, MLZ, MLY, M3L, HYP) alongside explicit CCD
   entry. Preserve Expert JSON rather than reconstructing it from the form.
2. Track endpoints against stable entity/copy identities and translate chain
   labels only at serialization. Sequence/type/copy edits invalidate prior
   validation and surface affected annotations for repair/review; never
   silently retarget or discard a modification or bond. Existing saved drafts
   receive empty annotation defaults. No additional state/UI dependency.
3. Implement bounded model-native CPU preflight below, with both Regular and
   Expert chemistry inputs using the same checks. Reject unsupported
   polymer–polymer bonds explicitly. Preserve unknown input fields according
   to the established Expert contract and keep inline data path-safe.
4. Reuse the existing confirmation and owner-scoped validation lifecycle.
   Show sites/components/bond endpoints for review, not just counts. Bind
   success to input and deployed native environment; deployments without the
   preflight contract cannot silently admit unverified chemistry.
5. Verify native preset fixtures, incorrect CCD/atom/site errors, custom-CCD
   process isolation, input edits/copies/reordering, Expert preservation,
   preflight timeout/busy paths and exact validation/submission identity.
   Commit cohesive backend/frontend milestones with exact OpenAPI handoff.

No upstream model patch, automatic deployment, or paid prediction is included.

The API's installed uniaf3 schema checks entity IDs,
indices and SMILES restrictions but does not contain a full CCD atom table.
Biotite is a development dependency, not an existing API runtime dependency.
Advanced manual atom-name entry retains broad ligand support through the
accepted model-native CPU preflight. It must resolve actual CCD/userCCD
components and atoms, not merely repeat JSON parsing. Reuse the deployed
model environment's component definitions rather than installing a separate
potentially different chemical dictionary in the API runtime.

### Proposed preflight implementation

Reuse the existing Continue/retained-validation flow: bounded local
parse/normalization, CPU preflight, then publish the successful validation ID.
Perform the remote wait outside the global validation lock, rechecking local
capacity before publication. Reject invalid inputs with actionable field or
endpoint details; timeouts/unavailable validation remain retryable and never
produce submit-ready input. The accepted UI timing is Continue, before the
existing confirmation page.

In the pinned AF3 environment, call `folding_input.Input.from_json`, construct
`chemical_components.Ccd` with inline userCCD, explicitly check referenced
component existence and unsupported/self/duplicate bonds, then call
`Input.to_structure(ccd)`. Native structure construction checks explicitly
bonded atom names against the effective CCD, including PTM replacements.
Retain the service prohibition on input filesystem paths and validate inline
userCCD as content before any native parser can interpret it as a filename.

This builds a minimal atom skeleton, not a prediction. No MSA search, templates
search, model weights, GPU allocation or conformer generation is required.
Run the native check in an isolated subprocess: the pinned `Ccd` constructor
mutates a shared cached dictionary when applying userCCD, so warm-process
reuse could leak custom definitions between requests. Bind successful checks
to the exact normalized input and selected deployment/CCD identity; a changed
target must not reuse stale validation. Bound preflight concurrency, input
size, runtime and returned errors.

This checks resolvable model input, not biological plausibility, complete
chemical valence, future conformer success or inference success. A CPU Modal
call still has startup latency and cost; no call has been made during design.

Evidence: pinned `common/folding_input.py:1408`,
`structure/parsing.py:649,718`, `constants/chemical_components.py:73,82`;
current service `alphafold3/router.py:313` and `validation.py:168`.
