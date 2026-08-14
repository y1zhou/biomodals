# Python Runtime Compatibility

Use this shared reference with both the Biomodals app-development and
workflow-development skills whenever a Modal image runs an older Python version
than the repository itself. The repository's `requires-python` value governs
local development and package installation; it does not change the interpreter
inside an app-owned or workflow-owned Modal image.

## Establish the runtime closure

Before editing remote code:

1. List the Python version of every image that can execute the changed code.
   Include app functions, workflow-native functions, included dependency apps,
   pull workers, and execution coordinators.
2. Trace the Biomodals modules imported by every decorated function or class in
   each image. The oldest interpreter in that import closure sets the source
   compatibility floor.
3. Confirm that the image contains the complete local import closure.
   `patch_image_for_helper(...)` adds shared helper, config, schema, and
   execution modules. Add app- or workflow-owned packages explicitly with
   `.add_local_python_source(...)`. A successful local import does not prove
   that the remote image contains the module.

Keep this analysis image-specific. An app may legitimately use Python 3.10
while its coordinator or containing workflow uses Python 3.13.

## Support the oldest interpreter

Check both parsing and runtime behavior:

- Avoid syntax introduced after the image's Python version. In particular,
  Python 3.10 and 3.11 cannot parse the Python 3.12 `type Alias = ...` syntax.
  `from __future__ import annotations` delays annotation evaluation but cannot
  make newer syntax parse on an older interpreter.
- Check standard-library availability separately from syntax. APIs such as
  `enum.StrEnum` and `datetime.UTC` are unavailable on Python 3.10. Put a
  version-gated standard-library, backport, or equivalent fallback in the
  shared module that owns the abstraction; do not copy app-specific fallbacks.
- Keep annotations imported by Modal-decorated functions and classes compatible
  with the defining image. Modal imports those modules before it can invoke the
  function.
- Treat parser-compatible code as unverified until the older-version import or
  fallback branch has been exercised. `ast.parse(..., feature_version=...)`
  catches syntax differences, not missing standard-library names or dependency
  incompatibilities.

## Resolve dependencies deliberately

Do not install the Biomodals package itself into an image whose Python version
does not satisfy the repository's `requires-python`. `patch_image_for_helper`
mounts the required source modules and installs their declared dependencies
instead.

For old-runtime images:

- Add environment-marked backports to project dependencies when shared modules
  require them.
- Use `ignore_dep_versions=True` only when the repository's pinned dependency
  versions cannot resolve for that Python version. It removes version
  constraints, so verify the versions resolved in the image.
- Use `skip_deps` only for dependencies that are absent from the image's runtime
  import closure or are installed separately by that image.
- Keep the reason for either exception beside the image construction. Do not
  weaken dependency constraints for newer images globally.

## Compose workflows without changing the floor

Including an app in a workflow deployment does not make the app's functions run
under the workflow image's Python version. Each decorated function retains its
own image, while workflow-native functions and the coordinator use their
declared workflow images.

Check all of those images because shared schemas, execution contracts, function
annotations, and workflow Node classes can cross their import boundaries.
Source-backed `--development` mode stabilizes package-qualified imports; it does
not change image Python versions or make incompatible dependencies compatible.

## Verify before remote execution

Extend `tests/app/test_execution_image_contracts.py` when an app or workflow
introduces an older interpreter or expands its imported source closure. Cover:

- every app-owned package required by the image's decorated functions;
- `ast.parse` at the oldest image Python version for each relevant local module;
- runtime execution of version-gated imports and fallbacks; and
- dependency-marker behavior in `patch_image_for_helper` when it changes.

Run these local contracts before any Modal smoke test. A change is ready only
when every modified remote entrypoint has a known image version, its complete
local import closure is mounted and parseable at that version, and its runtime
APIs and dependencies are available there.
