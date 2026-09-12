# AlphaFold3 input feedback and structure viewer

Status: implemented and verified offline; not deployed

Researched: 2026-09-11

Decisions updated: 2026-09-12

## Scope and conclusion

The requested experience is feasible with a focused frontend integration and a
Tool-specific result read API. Mol* already exposes the sequence strip, panel
composition, representation controls, confidence coloring and viewport styling
needed here. A separate molecular-viewing service, public structure upload,
Mol* fork, or new scientific execution is not required.

The following requirements are settled by the request:

- Expert JSON uploads must visibly acknowledge the selected input, summarize
  its entities and offer a bounded read-only preview without converting modes.
- The Job detail page must display the highest-ranked prediction using Mol*,
  without its left upload/database panel, while retaining a top sequence view
  and right-hand style controls.
- PAE plots must appear below the structure viewer. This website targets
  large screens only; mobile layout work is explicitly out of scope.

The user approved the consolidated implementation scope on 2026-09-12.
Existing scientific ranking, execution, ownership, input validation and archive
contracts remain unchanged. This document consolidates research, decisions and
implementation evidence rather than duplicating them in a research file.

The inspected baselines were backend `62be239` and frontend `93b295b`. Both
repositories implement this scope on `feat/alphafold3-ux`; the frontend's
unrelated README edit was preserved. Mol* was checked against
release **5.11.0**, commit
`7fc2ec55517e3da840ffe9fb09dab7d3065efec2`, and current npm metadata, not only
unversioned examples.[^release] AF3 source was inspected at the app's pinned
upstream commit `8f8abfedb88024c631f641e9f8a282e50afb7146`.

## 1. Expert upload baseline and required changes

At the inspected baseline, the file was read into `expertJson` and
`expertFilename`; its seeds
also update the shared seed field. However, the Expert card contains only a
file-drop control, whose selected filename is rendered in muted text. There is
no explicit reading/loaded state, JSON preview or parsed entity summary on
that screen. The fuller preview exists only after Continue initiates server
validation.[^input-ui]

Local acceptance checks JSON syntax, a single JSON object and the seed array.
It is not native AF3 validation. A successful local read should therefore say
"JSON loaded" rather than "Valid AlphaFold3 input". Server validation remains
the authority for native entities and scientific settings.[^input-state]

There are two additional state hazards worth fixing with this interaction:

- While a replacement is reading, the previous JSON remains submittable.
- A failed replacement leaves the previously loaded JSON in the draft, without
  an explicit explanation that the previous file is still selected.

The current latest-selection-wins read guard and Clear invalidation should be
retained. Add an explicit pending state that blocks Continue; on a failed
replacement, make the retained input unmistakable and require the user to
resolve the replacement before continuing. Do not silently submit older
contents under an error about a newer file.

### Regular and Expert are not interchangeable representations

The draft stores separate Regular `entities` and Expert `expertJson` fields.
Switching modes changes only `mode`; it does not convert either input. Name,
seeds and prediction settings are shared. Upload overwrites seeds, but does not
populate the Job name. At validation time the visible Job name and seed fields
override the corresponding JSON fields.[^input-state]

The Regular editor can represent protein, DNA and RNA sequences, copies, and
simple CCD/SMILES ligands. Native JSON can additionally contain chain-ID groups,
modifications, explicit bonds, MSAs, templates and custom CCD definitions.
Converting arbitrary JSON into Regular fields can remove scientific
information or change identifiers.

Even an apparently empty field can matter: an explicit empty MSA disables
that search, and `templates: []` suppresses template searching. Omitted fields
have different semantics. Existing preview `advanced_counts` uses truthiness,
so zero counts are not evidence that conversion is lossless.[^input-backend]

**Accepted:** show a clear loaded-file message, compact entity summary and an
expandable, bounded, scrollable, read-only JSON preview in Expert mode. Keep
the uploaded document authoritative for Expert submission; retain the existing
server confirmation step. Do not introduce automatic bidirectional conversion.
A future explicit "Import into Regular" action would need a documented,
lossless subset and a warning for every unsupported field.

The preview should render escaped text, not HTML. Avoid an IDE/editor
dependency for a read-only preview. Large inline MSAs/templates can make a
pretty-printed document expensive: calculate display content only when its
source changes, limit the visible preview, disclose truncation, and preserve
the complete original for validation. Do not mount an unbounded syntax tree.

**Accepted:** fill a blank Job name from the document name while preserving a
name the user has already entered. Continue importing the document seeds into
the visible seed control. Explain that the visible name and seeds override
their document fields; the raw preview is not a promise of byte-for-byte
submission. Inline JSON editing is not part of this change.

## 2. Which prediction is the top model?

This is already a backend scientific contract, not a new browser ranking:

1. Consider every sample for exactly the Job's normalized requested seed set.
2. Sort by `ranking_score` descending, then seed ascending, then zero-based
   sample index ascending.
3. Use the immutable request manifest's `best`, which must equal ranking row
   zero.[^best]

The accumulated Inference Run Summary is the wrong source: compatible requests
share a scientific run root, which may later acquire additional seeds. The
viewer must resolve the Job's exact Inference Request View. It must not glob
for a plausible CIF or recompute the winner from rounded confidence JSON.
Upstream summary scores are rounded to two decimals; the ranking manifest
retains the ranking values used for selection.[^best][^confidence-code]

The actual published artifact roles are:

| Role | Viewer use |
| --- | --- |
| `request_best_model_cif` | Native coordinates for the selected prediction |
| `request_best_summary_confidences` | Small summary metrics, if shown |
| `request_best_confidences` | Source for the PAE plot; avoid sending unrelated contact probabilities and atom arrays |

`request_archive_member_for_role()` already resolves these roles to the
presentation filenames. Its synthetic test role `request_best_model` is not
the publisher's actual CIF role; new tests should use real publication
fixtures.[^roles]

The UI should call this the "Highest-ranked prediction", identify seed and
sample, and explain that it is highest-ranked **within this Job**. AF3's ranking
score combines confidence with clash/disorder terms; it is neither a percent
confidence nor proof of a correct interaction.[^af3-output]

AF3 request publication currently requires all requested seeds and samples.
The generic service supports partial Results for other Tools, but that does
not establish a partial-AF3 viewer contract. Do not show a model from an
unfinished/failed AF3 request merely because a seed file exists. A completed,
published request is the first-release eligibility boundary.[^best]

## 3. Mol* integration choices

### Prefer the configurable Plugin UI

Mol* offers a packaged `Viewer`, `PluginUIContext` with reusable React
components, a plugin without built-in UI, and lower-level Canvas3D. The
requested layout fits the second option: retain its tested molecular controls
but compose a smaller interface. Building controls directly on Canvas3D would
recreate structure state, selection and interaction behavior.[^instance]

The `Viewer` wrapper can already hide the left panel and log, preserve the
sequence view, and configure viewport buttons. However, it imports an
extension map whose default is all viewer extensions. Disabling an extension
at runtime does not establish that the frontend bundle omits its code.
`DefaultPluginUISpec()` plus explicitly selected features gives a clearer
integration boundary.[^viewer-options][^viewer-spec]

Use one lazy-loaded AF3 result component, containing a small
Mol*-owning component with a locally defined UI spec. Reuse Mol*'s own top
sequence and selected right-side components. No public iframe, additional
Mol* web component framework, global `window.molstar`, or general-purpose
molecular-viewer abstraction is needed.

The optional `@molstar/molstar-components` package is not needed: its stable
line is Preact-based and its React migration is experimental and oriented
toward MolViewSpec editors/state builders.[^components-package]

### Verified control surface in Mol* 5.11.0

| Requested behavior | Exposed API | Important distinction |
| --- | --- | --- |
| Embed within the Job page | `layout.initial.isExpanded: false` | Expanded mode is a separate page-covering layout |
| Retain sequence and style UI | `layout.initial.showControls: true` | Setting this false hides all layout controls, including sequence |
| Remove the left panel | `components.controls.left: 'none'` | Unlike collapsing, this removes the component from the layout |
| Remove the bottom log | `components.controls.bottom: 'none'` | Application error/loading states must remain visible |
| Keep the top sequence strip | Default top `SequenceView` | `sequenceViewer.defaultMode` and `modeOptions` support chain, polymers and everything |
| Curate the right panel | `components.structureTools` | Retains Mol*'s scroll wrapper; use `controls.right` only when replacing the whole region |
| Remove remote snapshots UI | `components.remoteState: 'none'` | Not a network sandbox by itself |
| Disable file-drop import | `components.disableDragOverlay: true` | Hiding the left panel alone leaves a drag-import interface |
| Disable density streaming | `PluginConfig.VolumeStreaming.Enabled: false` | No density service is needed for an AF3 prediction |
| Keep reset, screenshot and fullscreen | `PluginConfig.Viewport.ShowReset`, `ShowScreenshotControls`, `ShowToggleFullscreen` | Independent from layout panels |
| Remove irrelevant controls | `ShowAnimation`, `ShowTrajectoryControls`, `ShowXR` | For one static model; XR accepts `'never'`, not a boolean |
| Match canvas/background colors | `canvas3d.setProps()` or `PluginCommands.Canvas3D.SetSettings` | UI CSS does not change the WebGL renderer's background |

These are source-level extension points, not proposed CSS selectors that hide
otherwise active interfaces.[^ui-spec][^ui-layout][^config]

### What "right-side style controls" can include

The stock `DefaultStructureTools` contains source/model controls,
measurements, superposition, Quick Styles, procedural animation, component
controls, volumes and extension controls. It is broader than the requested
style panel.[^right-controls]

Useful reusable pieces are:

- `StructureQuickStylesControls`: default/cartoon/spacefill/surface presets
  and default/illustrative appearance.
- `StructureComponentControls`: component visibility, representations and
  color/size themes.
- `StructureMeasurementsControls`: optional distances, angles and dihedrals.

**Accepted:** use Quick Styles and component controls, with measurements in a
collapsed section. Omit source selection, superposition, procedural animation,
volumes, remote state and imports. Preset changes can reset
coloring, so verify their interaction with the chosen confidence theme; do
not leave a pLDDT legend visible after switching to chain coloring.[^quick-styles]

The sequence panel already connects structure hover/selection with residue
display. Chain and residue interactions can be controlled through Mol*'s
selection/focus managers if later needed; synchronizing another independent
sequence widget is unnecessary.[^selections]

## 4. Confidence coloring needs precise wording

AF3 native CIF provides both per-atom pLDDT in `_atom_site.B_iso_or_equiv`
and ModelCIF quality-assessment categories. Its local ModelCIF value averages
the atoms in each residue. The global value is an average over atoms, not an
unweighted average over residue means.[^af3-cif]

Mol*'s `plddt-confidence` theme prefers mapped ModelCIF local scores and falls
back to atom B-factor values where a local score is unavailable. Its
quality-assessment extension registers the theme, presets and tooltips; its
default local tooltip refers to the residue metric. Therefore, describing
every displayed value as "atom pLDDT" would be wrong.[^plddt][^qa]

Nonpolymer CIF entries can have `label_seq_id='.'`, which complicates
residue-based QA mapping. The Mol* property reader maps QA entries through
residue indices; source inspection alone does not establish correct mapping
for every ligand/custom CCD. A protein-only browser fixture would miss this
problem.[^qa-prop][^af3-cif]

**Accepted:** initialize polymers with the existing residue-confidence theme
and ligands with element coloring, with chain coloring available as an
alternative. Preserve scientific colors even though the surrounding website
is monochrome. Use a short visible confidence legend and explain the
residue-average meaning. Do not invent a custom confidence aggregation or
silently label the B-factor field as experimental displacement.

Mol*'s confidence buckets are orange for scores up to 50, yellow through 70,
light blue through 90 and dark blue above 90. Negative/missing-theme cases
need truthful fallback treatment rather than a fabricated zero confidence.
Tests should cover the exact boundary values and the source of tooltips.[^plddt]

**Accepted:** show seed, sample, native pTM, nullable ipTM, ranking score and a
clash warning above the viewer. Preserve nulls; a monomer's unavailable ipTM is
not zero. The user additionally requested PAE plots below the viewer. Use the
same highest-ranked prediction for both structure and PAE; other-model
selection remains outside this change. The implemented PAE interaction and
bounded delivery are detailed below.[^confidence-code][^af3-output]

### PAE below the viewer: additional research

The native best-sample confidence JSON contains `pae`, `token_chain_ids` and
`token_res_ids`, alongside unrelated contact probabilities and atom arrays.
For a row-major heatmap, `pae[i][j]` belongs at horizontal coordinate **j**
(scored token) and vertical coordinate **i** (alignment-frame token). Preserve
both matrix triangles; PAE need not be symmetric. Missing values must remain
distinct from low error.[^confidence-code][^af3-output]

Token indices, not residue IDs, uniquely identify matrix positions. Standard
polymers normally contribute one token per residue, whereas ligands and many
modified residues contribute atom tokens with repeated chain/residue IDs.
Derive chain boundaries from the output token order, not submitted sequence
lengths. The confidence JSON does not provide token atom names or the native
frame-validity mask; it cannot establish an exact atom selection merely from
a repeated chain/residue pair.[^pae-tokens]

The pinned model's PAE bins extend to a final center of 31.75 Angstroms, and
the output writer stores PAE at one decimal place, with nonfinite missing
values represented as null. There is no `max_predicted_aligned_error` field.
A fixed 0–32 Angstrom scale is appropriate for this version; its legend should
say that lower error is better.[^pae-bins][^pae-serialization]

Mol* exports `MAPairwiseScorePlot`, but it is not a drop-in native AF3 plot:

- Its metric and AFDB example use residue indices, which cannot faithfully
  represent expanded ligand/modified-residue token grids.
- Its draw loop traverses the full matrix and creates an N-by-N canvas before
  encoding an image; the displayed plot size does not bound that work.
- Its drag interactions overpaint the structure, changing the agreed
  confidence/element coloring. The public component has no simple prop that
  disables that behavior.
- Its coordinate mapping needs explicit adaptation; do not inherit axis
  semantics from an AFDB example or fetch public AFDB data for this Job.

Reusing Mol* for the structure does not require using this particular PAE
component. A token-faithful image/canvas beneath it can be smaller and clearer
than converting AF3 data to an incompatible residue model.[^pae-molstar]

Size is material even on desktop. At 5,120 tokens, one matrix has 26,214,400
cells: 100 MiB as float32, before JSON or renderer overhead. The service's
5,120-polymer-residue admission limit is not a model-token limit, because
ligands/modifications can expand. Sending only PAE instead of the whole
confidence object avoids a second quadratic contact matrix, but does not by
itself bound PAE transfer or server parsing.[^pae-limits]

**Accepted:** large matrices may use a clearly labeled reduced-resolution
overview, with complete native values retained in the existing download.
Its sampling or aggregation rule must be documented; overview pixels must
not masquerade as exact token-pair measurements.

**Accepted:** provide a hover popup with chain IDs, residue names/indices and
associated confidence values, without linking the plot to the sequence or
structure display. Include drag-rectangle zoom and Reset alongside hover.
Use a bounded numeric overview/window resource rendered
with browser Canvas, plus ordinary HTML/SVG labels and pointer events. This is
a contained, moderate cross-repository change, not a scientific-model change.
Correct token mapping, bounded data access and async request handling are more
work than drawing the heatmap.[^pae-browser]

**Accepted hover behavior:** a reduced overview describes the displayed token
ranges and reduction. After zooming into a bounded full-resolution window,
hover reports original token indices, chain/residue labels and exact stored
PAE to one decimal place. Hover reads already-loaded window data; committing a
zoom fetches a new bounded window. Do not add stationary-hover requests to
retrieve exact pairs from a reduced overview. No structure/sequence linking is
included in this release.

**Accepted overview reduction:** color each displayed block by the arithmetic
mean of its available native PAE cells, preserving the separate X/Y ranges
and asymmetry. Show that it is a block mean and disclose missing cells; an
all-missing block remains unavailable. This summarizes the region but can
smooth isolated high errors, so exact measurements require a full-resolution
window. Label the aggregate "Block mean PAE" rather than implying that a pixel
is an exact token-pair measurement.

**Popup content:** show the aligned/frame token on Y and the scored
token on X, each with chain ID, native residue index and residue/CCD name; show
their directional PAE in Angstroms. Include each residue's mean pLDDT as local
confidence, explicitly distinguished from pairwise PAE. Do not present global
pTM, ipTM or ranking score as residue-level values. Null confidence remains
unavailable, never zero. Repeated ligand/modified-residue labels must also show
the original token index; do not invent a token atom name. Reduced overview
cells describe ranges, not a single invented residue pair.

Reuse the already parsed Mol* CIF for names and residue confidence, not a new
backend CIF parser. In the pinned AF3 writer, token residue IDs and output
`auth_seq_id` derive from the same internal residue IDs. Resolve the token's
label chain/entity, then use Mol*'s ligand-safe `findResidue` with `auth_seq_id`;
do not assume `label_seq_id` exists for ligands or use array position as a
residue ID. Read `label_comp_id` from the matched residue and average
`B_iso_or_equiv` over its actual atom segment. This is a residue-mean pLDDT,
not the individual expanded token's atom score. If the same-prediction CIF
is unavailable or cannot be matched, retain the native token IDs and PAE and
mark the additional annotation unavailable.[^pae-annotations]

This numeric-grid approach needs no additional frontend plotting/zoom library
and can avoid a backend PNG encoder. Mol* is still the new dependency for 3D.
Native Canvas/pointer events supply the interaction primitives, not a complete
ready-made scientific chart; labels, coordinate mapping, null handling and
tests remain application work.[^pae-browser]

**Accepted dependency constraint:** reuse the existing `orjson` parser; do not
add a streaming parser, plotting library or image encoder for PAE. Mol* remains
the separately requested dependency for the structure viewer. Pass JSON bytes
directly to `orjson.loads` without an intermediate UTF-8 string copy. It builds
the full Python object, including the unused quadratic `contact_probs` array;
discarding that field afterwards does not avoid peak parsing memory. It also
holds the GIL during parsing, so a thread alone cannot promise uninterrupted
service responsiveness.[^pae-orjson]

**Preparation:** enforce a decompressed confidence-member byte limit before
reading/parsing. The existing single artifact worker bounds preparation and
window reads. Reuse an archive-identity-bound float32 matrix, validate
token/matrix dimensions and release unused contact data. The AF3-local reader
retains at most two previews in memory, evicting before another preparation
would exceed that count. Every access still requires a fresh verified archive
lease; removing an archive immediately prevents serving a retained preview.
Bounded memory entries may remain until eviction or process exit and can be
reused if the identical archive is restored. There is no second disk cache,
worker pool, or provider download implementation.

**Accepted over-limit fallback:** when the confidence member exceeds the
measured safe parsing limit, show an explicit "PAE preview unavailable"
explanation while retaining the valid structure, summary and native downloads.
This affects only the preview, not scientific execution or Job success. Keep
other PAE loading/failures independent in the same way.

## 5. Fit the website without forking Mol*

The frontend uses React 19.2.7, Vite 8, Tailwind 4 and Geist. Its established
visual direction is a light, monochrome shell; the shared Job page currently
has a `max-w-5xl` container. There is no existing molecular-viewer dependency.
An existing lazy `HumanizationResults` mount is the natural Tool-specific
integration precedent.[^frontend-shell]

Mol* ships light/dark/blue skins. Its skin scopes normalization and layout
under `.msp-plugin`; configurable Sass colors include background, foreground,
hover and current-entity accents. Some geometry variables, including panel
widths, are ordinary Sass variables rather than `!default` configuration
points. Do not assume every skin variable can be overridden with Sass
`with`.[^skin][^skin-vars]

Keep Mol*'s base skin and use a small viewer-scoped stylesheet for
Geist, neutral control backgrounds, borders, focus rings and panel sizing.
Set the canvas background through the renderer API. Use website cards/buttons
outside the viewer. Add Sass only if changing the compile-time palette proves
cleaner than a handful of scoped overrides; measure the result before choosing.
There is no reason to rewrite every internal control as a website component.

Desktop shape:

```text
Job title and existing status/download actions
Highest-ranked prediction · seed … · sample … · ranking score …
┌─────────────────────────────────┬───────────────────────┐
│ Mol* sequence strip             │ Style controls        │
├─────────────────────────────────┤ Representations       │
│                                 │ Colors / visibility   │
│ Native predicted complex        │ Optional measurements │
│                                 │                       │
└─────────────────────────────────┴───────────────────────┘
Confidence legend / viewer loading or failure message
PAE plot for the same highest-ranked prediction
Existing execution stages and timings
```

**Accepted:** this is a large-screen-only website; do not implement a mobile
layout or mobile-specific tests for this work. Keep sequence above the canvas
and style controls on its right. Mol*'s reactive layout follows viewport
orientation/width rather than the size of our containing card, so choose an
explicit desktop layout and test its actual embedded width, page/sidebar
resizing and fullscreen behavior.[^layout-skin]

## 6. Private, bounded structure delivery

### Existing delivery infrastructure

Before this change, AF3 exposed retained input-document reads and whole-result
archive downloads, but no structure/result metadata endpoint. Finalization
builds a verified local `.tar.zst` Request Retrieval Archive from the exact
invocation publication. Prepared results retain filename, size, SHA-256 and
archive schema. Owner lookup conceals other users' jobs with 404.[^service]

The shared flow is a CSRF-protected `prepare-download` POST followed by an
authenticated GET. `ArtifactCache` owns integrity verification, restoration,
leases and explicit admin cleanup. It memoizes verified archive fingerprints;
unchanged reads do not rehash the archive on every request. The viewer should
reuse this boundary, not introduce a second independent download/cache
system.[^cache]

### Selected delivery boundary

| Option | Benefit | Cost / trade-off |
| --- | --- | --- |
| Read selected members from the existing leased archive | Reuses ownership, integrity, restoration and lifetime contracts; no new provider reads on a cache hit | `.tar.zst` requires sequential decompression; a cold cache may restore the full archive |
| Resolve the exact publication and retrieve only its CIF from the provider | Avoids downloading/decompressing unrelated outputs on the service | Needs individual-artifact caching, concurrency and immutable-result binding; more service integration |

Use the existing leased archive and fixed AF3 roles.
Never send the full archive to JavaScript just to display one structure. Keep
the member reader AF3-specific unless another Tool actually needs it. A
small companion metadata resource exposes seed, sample, ranking score,
prediction count and the accepted summary fields. The implemented HTTP
contract is recorded below.

Performance limits must be explicit. Current archives are sorted by member
name, so the submitted display name affects where the best-model alias lands.
Some names place all seed folders before that CIF. Historical service archives
have the same schema label but unsorted member order. Consequently, streaming
can stop once the required members are found, but a near-complete scan is
possible. Do not promise random access or assume the manifest comes first.
Capture required members in one pass where practical; do not decompress the
archive once for each metadata field.[^archive-reader]

The existing full archive inspector hashes every decompressed file and checks
the complete inventory. That is a publication/restoration check, not the
correct hot path for every viewer GET. A selective reader should use the
verified leased descriptor, bounded manifest/member parsing and existing role
resolution. Avoid filesystem extraction and arbitrary caller-supplied member
or Volume paths. Do decompression outside the async request event loop, with
bounded concurrency and cancellation cleanup.[^archive-reader][^cache]

The local unfavorable-order scan measurement is recorded below. Retain the
direct-artifact option only as a future alternative if real archive latency
justifies its extra integration. Do not repack historical Results: changing
bytes breaks their recorded immutable hash.

HTTP behavior follows the established humanization result pattern:
owner and Tool check, published-result eligibility, cache lease, coded cache
miss, and shared preparation followed by one retry. Return native CIF bytes
with owner-private response headers. Expose no internal provider paths or
credentials. A malformed/missing artifact must produce a clear viewer error,
not a blank canvas or a different prediction.[^humanization-reader]

### Browser data and network boundary

Use the same-origin API client to fetch the selected CIF with session
credentials and request cancellation, then feed it into
`builders.data.rawData`, `parseTrajectory(..., 'mmcif')` and a selected
representation preset. `loadStructureFromData()` exposes the equivalent
high-level sequence but its convenience signature does not include all preset
choices. Preserve the native multi-chain complex; do not silently load only
the first chain or apply an unrelated biological assembly.[^loaders]

Do not call `loadPdb`, `loadAlphaFoldDb`, remote snapshots, density or Model
Server functions for private predictions. Removing their UI is not proof of
zero network access: explicitly select extensions and verify a browser
network allowlist. Bundle the pinned library/assets with the website instead
of loading executable code from a public CDN.

The CIF must exist in browser memory for parsing/rendering; bounded delivery
does not mean zero-copy rendering. Avoid additionally retaining its raw text
in IndexedDB or a long-lived query cache, and release it with the viewer.
Keep native archive/CIF downloads available independently of WebGL.

## 7. Lifecycle, performance and compatibility checks

Mol* 5.11.0 declares Node >=22 and React/ReactDOM peers >=16.14. Those ranges
permit the current frontend, but are not proof of a working React 19/Vite 8
build. Its source also contains newer React view-model examples; examples are
not a substitute for checking initialization and cleanup.[^package][^react]

The website uses React StrictMode. Do not disable it to accommodate a copied
example. Own exactly one live plugin per mounted result, ignore stale async
initialization/fetch completion, abort pending requests, and dispose the
plugin on navigation/auth changes. If using `createPluginUI` with a separate
React root, retain and unmount that root too: the supplied `renderReact18`
helper creates a root without returning it. Rendering `<Plugin>` inside the
website-owned React tree is another supported option.[^react-root][^dispose]

**Accepted:** automatically load the viewer when a completed Job opens,
including shared result preparation when its archive is not cached. Show a
placeholder/progress state without blocking status, stages or downloads.
Load the viewer code only on eligible AF3 results. Do not recreate the plugin
on ordinary Job polling or a style change. Keep one structure loaded initially.
The production AF3 result chunk is approximately 931 KB gzip and is absent
from the initial shell. Browser coverage exercises the native-shaped fixture,
controls, navigation and WebGL fallback, including development StrictMode.
Production-scale structure parsing and browser peak memory remain unmeasured;
neither a small fixture nor npm's unpacked size establishes those costs.

WebGL failures must leave a readable error and downloads, not strand the
whole Job page. Exercise resize/fullscreen, context loss, navigation during
loading and keyboard focus. Inspect the actual production
CSP during implementation; no deployed headers were read for this research.
Do not weaken CSP speculatively or add server-side rendering/GPU dependencies
for a browser-only viewer.

## 8. Implementation and verification

The first round settled Expert presentation, imported name/seed behavior,
automatic result loading, curated controls, desktop-only layout and initial
scientific presentation. Reduced-resolution large-matrix overviews are also
accepted, together with residue-labeled hover, no sequence/structure linking,
no new PAE dependencies and an independent preview-unavailable fallback for
oversized confidence members. The final round accepted drag-rectangle zoom,
Reset, range hover on reduced overviews, exact hover in full-resolution
windows and explicitly labeled block-mean reduction. The user then approved
implementation across both repositories with offline verification only.

Completed implementation sequence:

1. Backend: add AF3-specific private result reads using existing ownership,
   publication, archive lease and preparation boundaries. Resolve one exact
   highest-ranked prediction for metadata, native CIF and bounded PAE windows.
   Measure source parsing and archive access locally to choose concrete byte,
   token, response and preparation-concurrency limits before enabling previews.
2. In parallel, frontend: implement Expert import feedback and the lazy Mol*
   component with the accepted desktop controls, styling and lifecycle. Keep
   the original input document and existing draft/validation behavior intact.
3. Coordinate exact offline OpenAPI types and deterministic result fixtures;
   connect the viewer and native Canvas PAE hover/zoom to that shared contract.
   Use the popup content described above, including clearly labeled
   residue-mean pLDDT, without extra parser or chart dependencies.
4. Run focused backend/frontend checks, real offline browser coverage and
   local memory/latency measurements. Record limits and evidence here, then
   hand off for review. No scientific jobs, deployment or production mutations
   are part of this implementation plan.

The endpoint/error schemas and operational bounds below implement these
decisions within existing service boundaries. The frontend uses the exact
offline OpenAPI export; it does not invent result fields independently.

Before release, verification should cover:

- Expert success/error/pending announcements, failed replacements, competing
  reads, Clear, draft recovery, and exact name/seed override behavior.
- Preservation of modifications, bonds, grouped IDs, empty MSAs and empty
  templates; no false success from local JSON parsing.
- Best-model identity across seeds/samples, exact and rounded-score ties,
  exclusion of unrelated cached seeds, and byte equality with the published
  best CIF.
- Owner/auth/Tool checks, cache miss/restoration, cleanup while leased,
  historical member ordering, malformed/oversized members and private headers.
- A real local CIF rendered in a browser with all external networking blocked,
  including protein complex, nucleic acid and ligand/custom-CCD cases.
- Mol* sequence/structure interaction, confidence legend/theme consistency,
  style controls, screenshot/fullscreen, desktop sizing, WebGL failure and
  StrictMode/navigation cleanup.
- Production bundling and lazy-chunk inspection; API type generation and
  focused offline service/frontend tests before any user-owned live test.
- PAE matrix orientation, asymmetric values, chain/token labels including
  ligand/modified-residue tokens, residue-name lookup, residue-mean pLDDT and
  loading/failure independence from the structure viewer. Include a large
  matrix in delivery/performance checks, including peak `orjson` parse memory
  and concurrent service responsiveness rather than response size alone.

Offline verification completed on 2026-09-12:

- Backend: all 1,682 tests pass, including 305 service tests and 25 AF3 result
  tests. Regression coverage includes presentation names prefixed by the
  canonical run name and private headers on preview errors. Pre-commit checks
  and focused production/result-test `ty` checks pass; existing fake-remote
  typing diagnostics remain in the shared browser/API fixtures.
- Frontend: all 95 unit tests and 27 browser tests pass, including seven AF3
  checks; lint and the production build pass. Two additional isolated Vite
  development StrictMode viewer/no-WebGL checks also pass.
- The authenticated offline fixture verifies exact best seed/ranking identity,
  nullable ipTM, native CIF rendering, directional PAE with nulls, and two
  expanded ligand tokens mapping to the same residue with mean pLDDT 70.

No production requests, provider operations, scientific jobs, deployments or
devserver restarts were performed. Rollout requires the updated API process
and frontend build; no database migration or scientific Modal redeployment
is needed solely for this UX change. Production CSP compatibility and
large-model browser resource use remain user-environment checks.

### HTTP preview contract

All paths below begin `/api/v1/alphafold3/jobs/{job_id}/prediction`:

| Path suffix | Response |
| --- | --- |
| none | `PredictionSummary`: immutable archive digest as `prediction_id`, selected seed/sample, exact ranking score, prediction count, nullable pTM/ipTM/clash, independent summary/PAE error codes and linear token IDs |
| `/model.cif` | Exact verified native bytes, `chemical/x-mmcif`, private/no-store |
| `/pae` | `PaeWindow`: matching prediction identity, original token edges, row-major values, valid-cell counts and `exact`/`mean` aggregation |

PAE queries accept zero-based `x_start`, `y_start` (default 0), exclusive
`x_end`, `y_end` (default token count), and `max_size` (1–512, default 512).
Each axis uses contiguous equal-width blocks except a possibly shorter final
block. Edge arrays contain both endpoints; `valid_counts` excludes nulls.
Values are serialized to three decimal places to bound aggregate response
size; exact native PAE remains at its original one-decimal precision.

Owner authentication is required; other-owner and other-Tool Jobs are 404.
Only succeeded AF3 Results with the established archive schema are eligible.
Unpublished/partial requests return `result_not_ready`; missing verified cache
returns `result_not_cached`. The client uses shared `prepare-download` and
retries once. Invalid artifacts return `result_invalid`; CIF/manifest/scan
limits return `preview_too_large`. Invalid rectangles return
`pae_window_invalid`. Metadata reports `summary_invalid`, `pae_invalid` or
`pae_too_large` independently; the PAE endpoint returns its matching code if
unavailable. All preview responses retain private/no-store headers.

### Resource limits and measurements

Code-owned preview limits, independent of scientific admission:

- Decompressed confidence JSON: 32 MiB; at most 2,048 tokens after parsing.
- Native CIF: 16 MiB; manifest: 4 MiB; summary: 1 MiB.
- Sequential archive scan: at most 4 GiB of decompressed members.
- At most 512 by 512 cells per response; only linear token metadata is sent
  with the prediction summary, never the full contact matrix.
- One active preparation/window computation on the shared artifact worker;
  at most two retained CIF/matrix entries. Cancellation waits for bounded
  worker completion before releasing its archive lease.

Local synthetic parser measurements on 2026-09-12 used three fresh processes
per size, JSON containing both native-shaped matrices, the production parser,
and a 5 ms event-loop probe while the parser ran in a worker thread. Peak RSS
is whole-process peak, including imports/source bytes, not retained matrix
memory or an isolated allocation count. These are local observations, not a
production latency guarantee; `orjson` still holds the GIL during parsing.

| Tokens | Source MiB | Median preparation ms | Maximum process RSS MiB | Maximum loop gap ms |
| --- | --- | --- | --- | --- |
| 512 | 2.18 | 28.4 | 138.2 | 12.1 |
| 1,024 | 8.70 | 95.5 | 235.8 | 47.4 |
| 1,600 | 21.22 | 229.1 | 423.0 | 107.5 |
| 1,800 | 26.86 | 297.2 | 505.6 | 153.4 |

A separate three-run archive probe placed 132.92 MiB of unrelated members
before the selected artifacts and the manifest last, with the 1,600-token
confidence source above. Median cold preparation was 275.69 ms, including
sequential decompression, selected-member integrity checks and parsing. A
400 by 400 window reduction plus JSON serialization took 24.99 ms median and
returned 1,393,894 bytes. A warm reader lookup took 0.004 ms median, excluding
authentication, archive-lease acquisition and HTTP delivery. The padding was
highly compressible (the entire archive was only 22,693 bytes compressed), so
these measurements do not characterize real archive disk/network throughput.

Provision memory for whole-document parsing in addition to the service's
ordinary runtime, concurrent authentication work and retained previews; the
32 MiB source cap is not a 32 MiB memory budget. Benchmark harnesses and
generated matrices remain in `/tmp`, not production source or committed data.

## Sources

External sources are upstream documentation/source, accessed 2026-09-11–12.
Mol* source links are release-pinned; local links record the inspected
implementation and may move as this branch develops.

[^release]: [Mol* 5.11.0 release](https://github.com/molstar/molstar/releases/tag/v5.11.0); [npm release metadata](https://registry.npmjs.org/molstar/latest), checked directly and returned 5.11.0.
[^input-ui]: Frontend [AlphaFold3SubmissionPage.tsx](../../../biomodals-frontend/src/pages/AlphaFold3SubmissionPage.tsx), upload handler lines 541–566, Expert card 718–732 and Continue 748; [FileDropZone.tsx](../../../biomodals-frontend/src/components/FileDropZone.tsx), filename lines 105–107.
[^input-state]: Frontend [alphafold3.ts](../../../biomodals-frontend/src/alphafold3.ts), draft type, regular serialization, local JSON parsing and name/seed overrides, lines 4–33 and 204–277; [AlphaFold3SubmissionPage.tsx](../../../biomodals-frontend/src/pages/AlphaFold3SubmissionPage.tsx), mode switch and retained validation flow.
[^input-backend]: [validation.py](../../src/biomodals/service/alphafold3/validation.py), native validation, path rejection and advanced counts; [msa_search.py](../../src/biomodals/app/fold/alphafold3/msa_search.py), `field_is_populated`; [input_enrichment.py](../../src/biomodals/app/fold/alphafold3/input_enrichment.py), template field-presence handling.
[^best]: [seed_predictions.py](../../src/biomodals/app/fold/alphafold3/seed_predictions.py), `ranked_rows` around line 1054; [request_results.py](../../src/biomodals/app/fold/alphafold3/request_results.py), request publication and manifest validation; [CONTEXT.md](../../CONTEXT.md), Prediction Ranking Order, Inference Request View and Partial Inference Request.
[^confidence-code]: Pinned AF3 [confidence_types.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/confidence_types.py), summary serialization and full confidence arrays; [confidences.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/confidences.py), unavailable interface TM score.
[^roles]: [request_results.py](../../src/biomodals/app/fold/alphafold3/request_results.py), `_best_artifacts` around line 451 and `request_archive_member_for_role` around line 934; [production contract tests](../../tests/app/test_alphafold3_production_contracts.py), synthetic role around line 3466.
[^af3-output]: AF3 [output documentation](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/docs/output.md), ranking and confidence semantics.
[^instance]: Mol* [plugin instance documentation](https://molstar.org/docs/plugin/instance/), integration levels and individually reusable React UI components.
[^viewer-options]: Mol* [viewer options](https://github.com/molstar/molstar/blob/v5.11.0/src/apps/viewer/options.ts), defaults and independently configurable panels.
[^viewer-spec]: Mol* [viewer plugin spec](https://github.com/molstar/molstar/blob/v5.11.0/src/apps/viewer/plugin-spec.ts), option-to-component/config mapping and extension registration.
[^components-package]: Official [Molstar Components README](https://github.com/molstar/molstar-components) and [changelog](https://github.com/molstar/molstar-components/blob/main/CHANGELOG.md), stable Preact and experimental React lines.
[^ui-spec]: Mol* [PluginUISpec](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/spec.ts), layout regions, structure tools, drag overlay and sequence options.
[^ui-layout]: Mol* [plugin layout](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/plugin.tsx), actual conditional mounting and drop handling.
[^config]: Mol* [PluginConfig](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin/config.ts), viewport and remote-service options; [viewer state documentation](https://molstar.org/docs/plugin/viewer-state/), renderer and interaction changes.
[^right-controls]: Mol* [DefaultStructureTools](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/controls.tsx), constituent controls around lines 331–347.
[^quick-styles]: Mol* [Quick Styles](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/structure/quick-styles.tsx), representation and appearance presets.
[^selections]: Mol* [selection documentation](https://molstar.org/docs/plugin/selections/); [SequenceView](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/sequence.tsx), entity/chain options and display modes.
[^af3-cif]: Pinned AF3 [mmcif_metadata.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/mmcif_metadata.py), global/local pLDDT around lines 182–228; [structure_tables.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/structure/structure_tables.py), atom B-values and nonpolymer sequence IDs.
[^plddt]: Mol* [pLDDT confidence theme](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/model-archive/quality-assessment/color/plddt.ts), ModelCIF preference, atom fallback and exact color boundaries.
[^qa]: Mol* [quality-assessment behavior](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/model-archive/quality-assessment/behavior.ts), registration, local-metric tooltips and optional pairwise plot.
[^qa-prop]: Mol* [quality-assessment property](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/model-archive/quality-assessment/prop.ts), local metric parsing and residue-index lookup.
[^pae-tokens]: Pinned AF3 [features.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/features.py), token expansion around lines 173–267 and alignment frames around 2108; [model.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/model.py), output token IDs and unmasked full PAE.
[^pae-bins]: Pinned AF3 [confidence_head.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/network/confidence_head.py), 64 bins, `max_error_bin` and bin centers.
[^pae-serialization]: Pinned AF3 [json_serialize_pybind.cc](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/json_serialize_pybind.cc), native PAE precision, null serialization and field names.
[^pae-molstar]: Mol* [AFDB PAE example](https://github.com/molstar/molstar/blob/v5.11.0/src/examples/alphafolddb-pae/index.tsx), [plot renderer](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/model-archive/quality-assessment/pairwise/plot.ts) and [plot UI](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/model-archive/quality-assessment/pairwise/ui.tsx), residue mapping, full-grid rendering and overpaint interactions.
[^pae-limits]: [inference_inputs.py](../../src/biomodals/app/fold/alphafold3/inference_inputs.py), input entity/polymer-residue limits; pinned AF3 [pipeline.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/pipeline.py) and [featurisation.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/data/featurisation.py), default unbounded token count and bucket overflow behavior. Matrix byte counts are calculated as N squared times element width, not benchmark measurements.
[^pae-browser]: MDN [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) and [pointer capture](https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture), native graphics and drag-event primitives, checked 2026-09-12; frontend [package.json](../../../biomodals-frontend/package.json), existing React/TanStack Query dependencies. The no-chart-library conclusion is an implementation assessment, not a measured prototype result.
[^pae-annotations]: Pinned AF3 [model.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/model.py#L406), output token IDs; [atom_layout.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/model/atom_layout/atom_layout.py#L1077), structure rebuilding; [structure_tables.py](https://github.com/google-deepmind/alphafold3/blob/8f8abfedb88024c631f641e9f8a282e50afb7146/src/alphafold3/structure/structure_tables.py#L761), native residue/auth ID defaults; Mol* [atomic index](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-model/structure/model/properties/utils/atomic-index.ts#L88), ligand-safe `findResidue`; [atomic hierarchy](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-model/structure/model/properties/atomic/hierarchy.ts#L132), atom/residue columns and segments. The popup joins the same prediction's CIF and native token IDs; it does not establish token atom identity or a universal mapping for arbitrary CIFs.
[^pae-orjson]: Official [orjson deserialization documentation](https://github.com/ijl/orjson#deserialize), direct byte input, Python-object output and GIL behavior, checked 2026-09-12; [pyproject.toml](../../pyproject.toml), existing `orjson` dependency. Preparation bounds and reuse are implemented by the application, not supplied by the parser.
[^frontend-shell]: Frontend [package.json](../../../biomodals-frontend/package.json), [main.tsx](../../../biomodals-frontend/src/main.tsx), [index.css](../../../biomodals-frontend/src/index.css), [JobDetailPage.tsx](../../../biomodals-frontend/src/pages/JobDetailPage.tsx) and [frontend ADR 0005](../../../biomodals-frontend/docs/adr/0005-live-api-mvp-frontend.md).
[^skin]: Mol* [base skin](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/skin/base/base.scss) and [color variables](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/skin/base/_colors.scss).
[^skin-vars]: Mol* [skin measures and derived variables](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/skin/base/_vars.scss).
[^layout-skin]: Mol* [layout dispatch](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/skin/base/layout.scss), [landscape](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/skin/base/layout/controls-landscape.scss) and [portrait](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/skin/base/layout/controls-portrait.scss) geometry.
[^service]: [AF3 router](../../src/biomodals/service/alphafold3/router.py), [AF3 adapter](../../src/biomodals/service/alphafold3/modal.py), [shared job delivery](../../src/biomodals/service/jobs_api.py) and [HTTP contract](../../src/biomodals/service/http_contract.py).
[^cache]: [ArtifactCache](../../src/biomodals/service/artifacts.py), acquisition/fingerprint verification and cleanup; [tool_runtime.py](../../src/biomodals/service/tool_runtime.py), exact archive restoration contract.
[^archive-reader]: [request_results.py](../../src/biomodals/app/fold/alphafold3/request_results.py), archive inspection around line 1199 and tar construction around line 1440. Historical service baseline `390c658` used the same `alphafold3-request/1` label before deterministic tar ordering in `f9f7106`.
[^humanization-reader]: [humanization/router.py](../../src/biomodals/service/humanization/router.py), owner/Tool checks, result eligibility and leased result reads.
[^loaders]: Mol* [loaders](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/plugin/loaders.ts), data versus URL/database loading; [hierarchy presets](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-state/builder/structure/hierarchy-preset.ts).
[^package]: Mol* [package metadata](https://github.com/molstar/molstar/blob/v5.11.0/package.json), Node engine, React peers and optional server-side rendering dependencies.
[^react]: Mol* [React examples](https://github.com/molstar/molstar/blob/v5.11.0/src/examples/react/index.tsx) and [UI view-model hook](https://github.com/molstar/molstar/blob/v5.11.0/src/extensions/plugin/hooks/use-ui-view-model.ts).
[^react-root]: Mol* [createPluginUI](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/index.ts) and [renderReact18](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin-ui/react18.ts).
[^dispose]: Mol* [PluginContext disposal](https://github.com/molstar/molstar/blob/v5.11.0/src/mol-plugin/context.ts), animation, managers, state and canvas teardown.
