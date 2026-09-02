# CUMAb app research

Research date: 2026-09-02

Possible implementation target: `src/biomodals/app/design/cumab_app.py`

## Recommendation in brief

CUMAb adds a genuinely different scientific axis to the planned stack: it
grafts parental CDRs onto thousands of complete human germline frameworks and
ranks the resulting structures with Rosetta energy and CDR-geometry criteria.
It is therefore aimed at preserving stability, expression, and binding
geometry while increasing humanness, rather than maximizing a repertoire-
learned sequence score
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/)).

Do **not** implement this as a small one-call wrapper around the repository.
The repository generates candidate sequences and example Rosetta command
lines, but explicitly leaves scheduling to the user and contains no code for
the paper's final filtering, energy ranking, or V-subgroup clustering
([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
[paper method](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/)). A
scientifically honest `cumab` app requires a durable multi-stage execution
plan, resumable fanout over more than 20,000 Rosetta jobs, and explicit upstream
patches. Licensing must be resolved before that implementation is distributed
or offered as a service.

## Immutable source and dependency pins

- Current upstream `main` is
  [`88cec68e89e81c2ec4a4c7499a324021062d4a83`](https://github.com/Fleishman-Lab/CUMAb/tree/88cec68e89e81c2ec4a4c7499a324021062d4a83),
  committed 2025-04-10. There are no tags, releases, Python package metadata,
  CI configuration, or automated tests in the repository
  ([repository tree](https://github.com/Fleishman-Lab/CUMAb/tree/88cec68e89e81c2ec4a4c7499a324021062d4a83)).
- Upstream pins Rosetta to Git commit
  [`d9d4d5dd3fd516db1ad41b302d147ca0ccd78abd`](https://github.com/RosettaCommons/rosetta/commit/d9d4d5dd3fd516db1ad41b302d147ca0ccd78abd).
  A later or current numbered Rosetta release is not an equivalent substitute
  without an oracle comparison
  ([installation instructions](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md)).
- The exact legacy environment is Python 3.7.11, Biopython 1.74, pandas 0.25.1,
  HMMER 3.3.2, and PyMOL 2.3.3 from `bioconda`, `conda-forge`, `schrodinger`,
  and `defaults`
  ([environment](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/CUMAb_environment.yml)).
  The code imports the removed `Bio.Alphabet` API, so it cannot simply be
  installed under Biomodals' Python 3.12 runtime
  ([PDB formatting module](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_pdb_format.py)).

CUMAb has no learned model weights. The repository pins an approximately
344-kB HMMER antibody-chain database in Git; its principal HMM SHA-256 at the
source pin is
`03c219baed350b7e0bcd80cd1dd90674eb4cd51dc184648507ebc2e59a19bf6f`
([HMM database](https://github.com/Fleishman-Lab/CUMAb/tree/88cec68e89e81c2ec4a4c7499a324021062d4a83/hmm_database)).
Its other scientific assets are the pinned Rosetta source/database and six
human amino-acid germline FASTAs: `IGHV`, `IGHJ`, `IGKV`, `IGKJ`, `IGLV`, and
`IGLJ`
([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
[database parser](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_graft.py)).

The germline instructions point to live IMGT/GENE-DB queries and give neither
a release nor checksums. Candidate identities and counts can therefore drift
as IMGT changes. At research time, the current downloadable reference
directory is release `202631-1` (27 July 2026), but reproducing the paper
requires the historical amino-acid FASTAs used by its authors, not today's
release
([current IMGT release](https://www.imgt.org/vquest/refseqh.html),
[CUMAb download links](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md)). Before implementation,
obtain or reconstruct the paper-era files; otherwise declare a Biomodals
scientific fork that pins six exact downloaded files, records their SHA-256
digests, and versions that identity into every cache and result.

## Licensing is an implementation gate

The CUMAb README says only “Licensed under the Non-Profit Open Software License
version 3.0”; the repository contains no license text or copyright notice
([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
[repository tree](https://github.com/Fleishman-Lab/CUMAb/tree/88cec68e89e81c2ec4a4c7499a324021062d4a83)). NPOSL-3.0 is a reciprocal
license with network-deployment/source-disclosure conditions and special
requirements on who may redistribute under that amended license
([official text](https://opensource.org/license/NPOSL-3.0)). Obtain maintainer
or legal confirmation before vendoring or modifying the scripts. The bundled
HMM database and XML protocols have no separate notices, so their redistribution
status is no clearer than the repository-level one.

Rosetta is not OSI open source. Its standard license permits internal
non-commercial use and forbids commercial/fee-based service use without a
separate University of Washington agreement; it also restricts distribution
of Rosetta and modifications
([official Rosetta license](https://docs.rosettacommons.org/demos/latest/LICENSE),
[download terms](https://downloads.rosettacommons.org/)). A public Modal image
must not redistribute Rosetta under an ordinary academic license. A viable app
needs a deployment-specific commercial/institutional entitlement and likely a
private build path or user-provided licensed artifact.

IMGT currently licenses data and metadata under CC-BY-4.0, while requests are
required for use of IMGT tools. The app needs only pinned data files, not the
hosted tools, and must provide attribution
([IMGT terms](https://www.imgt.org/about/termsofuse.php)). These independent
code, Rosetta, and data obligations should be documented in the app help and
image construction. This note is not legal advice.

## Scientific method

CUMAb begins with an experimental or predicted structure of a complete paired
Fv. It enumerates human heavy V/J and matching kappa or lambda light V/J
combinations, grafts the parental CDRs into each framework, and removes
frameworks containing `NG`, `N[^P][ST]`, or excessive cysteines. The paper
reports 63,180 theoretical kappa and 48,600 lambda combinations before
compatibility filters, usually leaving more than 20,000 candidates per
antibody
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/),
[generation source](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_graft.py)).

The default mode grafts complete CDRs under a custom CUMAb definition. `SDR`
mode instead retains specificity-determining residues, requires an antigen
chain, and restores antibody-antigen interface residues from the parent. Users
can also retain explicitly numbered parental residues. D genes are subsumed by
the fixed CDR-H3 rather than enumerated as framework components
([CLI arguments](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_args.py),
[grafting source](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_graft.py),
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/)).

For the structural stage, the paper relaxes the parental structure, threads
every candidate onto the lowest-energy relaxed structure, and applies Rosetta
side-chain packing plus constrained whole-protein minimization. Models are
ranked by the `ref2015` all-atom energy. Experimental-structure runs exclude a
candidate when any CDR C-alpha/carbonyl-O RMSD is at least 0.5 Angstrom; model-
structure runs intentionally do not apply that exclusion. Top designs are
then clustered by human V-gene subgroup to retain diverse low-energy
candidates
([paper method](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/),
[threading protocol](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/xmls/CUMAb.xml),
[RMSD script](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/find_RMSDs.py)).

“CUMAb score” must therefore not be presented as a humanness probability.
Human character arises by construction from human V/J frameworks; the ranking
quantity is Rosetta energy, accompanied by structural integrity and optional
interface metrics. The paper experimentally tested five antibodies and found
some selected designs retained parental affinity and improved stability, but
also says broader validation is needed
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/)).

## Supported inputs, species, and outputs

Upstream accepts exactly one PDB per working directory. It detects one heavy
and one kappa/lambda light variable domain with the bundled HMM database,
rewrites the Fv as light chain A and heavy chain B, and optionally rewrites one
antigen chain as C. A structure may be experimental or modeled; VHH and
unpaired chains are unsupported because the formatter and generator require
both heavy and light coordinates
([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
[formatter](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_pdb_format.py)).

The CLI allows `origin_species=human|mouse|rabbit`. Human and mouse share the
same motif-based CDR parser; rabbit uses a different H2/H3 expression. The
source itself warns that rabbit mode may be buggy and has not been validated
experimentally, so scientific support should initially be described as mouse
(and structurally compatible human) Fv, with rabbit explicitly experimental
([argument help](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_args.py),
[CDR parser](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_graft.py)).

The repository's two top-level scripts produce:

- `{name}_CUMAb_format.pdb`, `config.json`, extracted chain FASTAs, and
  temporary HMM/PDB artifacts;
- `{name}_grafted_sequences.csv` with `Germline combination` and one
  concatenated light-then-heavy `Sequence` column;
- example command files for the 15-run parental relax and per-candidate
  threading; and
- optionally `{name}_RMSDS.csv` with L1-L3/H1-H3 RMSDs.

Rosetta itself produces per-run PDB and `score.sc` artifacts. Upstream does not
produce the paper's final ranked/clustered result table
([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
[graft driver](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/CUMAb_graft_sequences.py),
[RMSD script](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/find_RMSDs.py)).

## Compute, storage, batching, and determinism

This is a CPU/HPC workload, not a CUDA workload. The paper reports that
designing and ranking roughly 20,000 candidates took a few hours on its
compute cluster. Upstream directs users to run the initial relax 15 times,
select the lowest-scoring structure, and invoke one Rosetta process for every
candidate, but supplies no scheduler
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/),
[README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md)).

Rosetta packing/minimization is stochastic and the provided flags set
`nstruct=1` without a random seed. Exact PDBs, energies, and possibly ranking
can therefore vary across runs; thread count, compiler, CPU architecture, and
Rosetta database/source identity are additional numerical identities
([flags](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/flags),
[protocols](https://github.com/Fleishman-Lab/CUMAb/tree/88cec68e89e81c2ec4a4c7499a324021062d4a83/xmls)). A Biomodals implementation
should derive and record a stable seed for every relax/candidate task. That is
a deliberate reproducibility patch, not bitwise equivalence to the unseeded
upstream instructions.

Candidate structures and score files can occupy many gigabytes, so neither
inputs nor results should be gathered into a single inline payload. Publish
each validated task result to a staged Volume, resume by content identity, and
return a compact ranked summary plus a `VolumePath` to structures. Benchmark
one relax and one thread job to size CPU, memory, timeout, scratch disk, and
fanout concurrency; upstream provides no per-process resource measurements.

## Validation, security, and upstream bugs

1. **The published XML files are syntactically broken.** `CUMAb.xml` has
   missing `<` characters on an `Add` and a `RotamerTrialsMinMover`, and
   `Relax.xml` has malformed `ScoreFunction`, `Reweight`, and
   `PreventResiduesFromRepacking` tags plus stray text. Patch files must be
   versioned, explained, and checked against the paper or maintainer before any
   equivalence claim
   ([CUMAb.xml](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/xmls/CUMAb.xml),
   [Relax.xml](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/xmls/Relax.xml)).
2. **Shell injection and path confusion are present.** User-controlled PDB
   paths and chain IDs are interpolated into `os.system` commands without
   quoting, and repository location is inferred by searching `sys.path` for a
   component named `CUMAb`. Never invoke those paths verbatim. Use argument
   arrays, immutable internal assets, safe generated filenames, and one fresh
   task directory per input
   ([formatting module](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_pdb_format.py)).
3. **Free-form regular expressions are unsafe.** `-screens` accepts arbitrary
   regexes and runs them repeatedly over tens of thousands of candidates.
   Expose only the published motif screens in v1; adding custom patterns would
   need length/syntax restrictions and timeout-safe matching
   ([arguments](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_args.py),
   [screening loop](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_graft.py)).
4. **PDB assumptions are brittle and destructive.** The formatter strips all
   hetero atoms, renumbers chains, concatenates extracted sequences, relies on
   motif regexes, and can abort on ordinary layout differences. Validate file
   size, atom/residue/chain counts, ASCII/PDB syntax, one heavy and one light
   variable domain, distinct optional antigen, canonical residues, and no
   unsupported models before scheduling
   ([formatter](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_pdb_format.py),
   [PyMOL cutter](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/pdbcutter.py)).
5. **Failure can be misreported as perfect geometry.** `find_RMSDs.py` catches
   every exception and writes six zeros, which is indistinguishable from exact
   structural identity. Fail that candidate with a diagnostic instead
   ([RMSD script](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/find_RMSDs.py)).
6. **The local release is scientifically incomplete.** There is no automatic
   selection of the lowest of 15 relaxations, no threading scheduler, no 0.5-A
   filtering policy conditional on experimental versus modeled input, and no
   energy ranking/V-family clustering. Implementing these is app-owned
   orchestration derived from the paper, not a transparent wrapper
   ([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
   [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/)).
7. **The README misstates one required filename.** Its six-name list repeats
   `IGKJ.fasta` where `IGLV.fasta` is required; the parser correctly expects
   `IGLV`
   ([README](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/README.md),
   [parser](https://github.com/Fleishman-Lab/CUMAb/blob/88cec68e89e81c2ec4a4c7499a324021062d4a83/scripts/modules_graft.py)).

## Equivalence strategy

First freeze a reference capsule: CUMAb commit, exact six IMGT amino-acid
FASTAs and hashes, bundled HMM files, corrected XML/flags with patch hashes,
Rosetta source/database commit, compiler/runtime, and legacy Python
environment. Preserve unmodified upstream behavior for formatting and sequence
generation where it is scientifically valid, while making security fixes at
the boundary.

For preparation/grafting, compare normalized formatted chains, detected
kappa/lambda type, CDR boundaries, parental/fixed positions, exact germline
combination set, motif exclusions, and exact generated sequences. Cover mouse
kappa and lambda, bound/unbound, CDR/SDR, explicit fixed residues, and modeled
versus experimental provenance. Rabbit should remain experimental until an
independent fixture exists.

For Rosetta, test fixed-seed structural and score fixtures rather than
expecting the unseeded command's bytes to match. Verify selection of the
lowest-energy parental relaxation, thread success, score parsing, experimental
input's per-CDR `<0.5`-Angstrom rule, modeled-input exception, energy ordering,
and V-subgroup clustering. Finally reproduce at least one published target's
reported candidate/design set from supplementary material. If the historical
IMGT data or exact paper ranking logic cannot be obtained, label outputs
“CUMAb-derived” and do not claim paper equivalence
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/)).

## Smallest workflow-compatible Biomodals boundary

The smallest honest public surface is one **humanize** operation for one PDB,
with typed `mode`, `origin_species`, optional antigen-chain ID, published motif
screens, explicit fixed residues, and modeled-versus-experimental structure
provenance. The operation should own one durable `ExecutionDefinition` with:

1. validate/format/generate candidates;
2. fan out 15 seeded parental relax tasks and select the lowest energy;
3. fan out one seeded threading task per candidate with resumable staged
   publication;
4. aggregate validated score/PDB outputs, compute CDR RMSDs, filter, rank, and
   cluster; and
5. publish a ranked CSV/Parquet, selected structures, normalized input,
   candidate/mutation provenance, failure table, and identity manifest.

This is an execution-kernel fanout/resumption app from its first faithful
version. A preparation-only operation could be implemented much sooner, but it
must be named `cumab-candidates` and must not be presented as completed CUMAb
humanization. Full output belongs on a Volume; only the compact summary should
be inline.

## Place in the antibody-humanization stack

CUMAb complements Sapiens rather than extending its scoring. Sapiens proposes
sequence-only, independent-chain substitutions learned from human repertoires;
CUMAb proposes wholesale human V/J frameworks around fixed parental CDRs and
uses paired-Fv structure/energy to select compatible frameworks. CUMAb can
therefore address framework-CDR and VH/VL structural compatibility that a
Sapiens residue probability cannot, but it is orders of magnitude more
expensive and its Rosetta license is deployment-limiting
([Sapiens/BioPhi paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/),
[CUMAb paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10842793/),
[Rosetta license](https://docs.rosettacommons.org/demos/latest/LICENSE)).

In the final workflow it should be an optional alternative candidate generator,
not a mandatory stage after Sapiens. Applying CUMAb's paper protocol to a
Sapiens candidate would replace that candidate's framework and would no longer
mean “structurally score the Sapiens design.” If a later workflow needs a
general Rosetta Fv scorer, design and validate that as a separate operation.
