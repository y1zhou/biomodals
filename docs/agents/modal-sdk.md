# Modal SDK compatibility

The project targets Modal 1.6.x, with 1.6.0 recorded in `uv.lock`. Minor SDK
releases can remove deprecated APIs, so review warnings and release notes before
moving the compatible-release constraint to the next minor version.

Use `Function.local(...)` for same-container calls and pass `Function.local`
when a workflow needs a callable. Remote execution continues to use the existing
execution driver. The catalog's use of `Function._raw_f_` is a narrow private-SDK
exception for original docstrings and signatures: `.local` exposes a generic
signature, and Modal has no public source-introspection replacement for the
removed `get_raw_f()`. Keep invocation code off this private property.

The 1.6.0 review also checked Modal classes for custom constructors, removed
object properties, Sandbox filesystem APIs, Sandbox scheduling assumptions, and
consumers of changed Modal CLI JSON output. No production migrations beyond
local function access were needed for those changes. Classes already use
`modal.parameter()` and `@modal.enter()`.

## HuDiff scientific environment

HuDiff-Ab workers reuse immutable validated image `im-08MH3kBTTZNGSTjyazY6EW`.
The current helper, config, schema, execution and HuDiff package sources are
mounted over that image's captured source. The coordinator still builds against
the current project dependencies. Modal injects the deployment SDK separately
from the scientific environment's installed distributions.

Rebuilding the worker from the old recipe during this SDK update would conflict
with `modal==1.5.5` in `runtime-constraints.txt` and invalidate the full environment
fingerprint. Preserve those historical constraints and the scientific identity;
do not accept a new digest merely to get a build to pass. A future replacement
image requires an audited dependency inventory and scientific validation as
described in [the HuDiff research report](../research/humanization/hudiff-ab.md).
The image must be accessible in the deploying workspace; there is deliberately
no fallback to an unvalidated rebuild. This also avoids silently changing Conda
builds when deploying this SDK update.

## Verification boundaries

Before changing an isolated adapter, account for missing SDK members, callable
signature loss, async/generator return behavior, stale test doubles, missing
remote source modules, incompatible Python versions, and image dependency or
fingerprint drift. Prefer existing behavioral flows over new API-shape tests.

Run the existing suite with `uv run --frozen pytest -q --junitxml=<report.xml>`
and capture app/workflow catalog and help output. Check task-image imports on
Python 3.10 and 3.11 as well as the project's Python 3.12. These checks exercise
local SDK integration; they do not prove cloud image availability, image builds,
mounted-volume behavior or scientific GPU results. A credentialed deployment
smoke run remains necessary to verify those boundaries.
