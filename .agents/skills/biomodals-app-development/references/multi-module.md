# Multi-Module Apps

Keep `*_app.py` as the discoverable composition root. Split when stages, caches,
or backends hide the Modal topology; keep short linear apps together.

## Recommended shape

```text
src/biomodals/app/score/tool_app.py       # AppConfig, images, decorators, CLI
src/biomodals/app/score/tool/
├── __init__.py
├── contracts.py                         # frozen plans and manifests
├── cache.py                             # identities and publication policy
├── backend_a.py                         # pure planning/parsing/reduction
└── postprocess.py                       # pure domain transformation
```

Keep in the composition root:

- `CONF`, images, volumes, `app`, decorators/resources, and the entrypoint.
- The high-level call graph, phase logging, and catalog-visible function exports.

Move behind narrow interfaces:

- Primitive immutable plans/manifests; cache identities and validation.
- Backend commands, parsers, reducers, and pure domain transformations.

Avoid a generic manager or service layer. Split by stable domain responsibility,
and let each module hide its file formats and upstream quirks.

## Modal source inclusion

Modal includes a Function's defining package, but not arbitrary imported local
dependencies. `patch_image_for_helper(...)` adds only shared Biomodals modules.

Explicitly include app-specific packages when runtime code imports them:

```python
image = patch_image_for_helper(base_image).add_local_python_source(
    "biomodals.app.score.tool"
)
```

Use `copy=True` only for build-time imports. A local import does not prove remote
image inclusion.

Prefer decorators in `_app.py`. Otherwise include, import, and re-export sibling
functions there; give Modal functions/classes unique names across included apps.

## Boundary checks

- Keep `Path`, files, Modal handles, queues, and volumes out of plans; cross
  boundaries with strings and primitives.
- Depend on narrow operations such as `plan`, `run_shard`, `validate`, and
  `merge`, not backend directories.
- Test pure modules directly; reserve `get_raw_f()` and fake `.remote()` handles
  for composition tests.
