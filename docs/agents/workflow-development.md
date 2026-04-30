# Biomodals Workflow Development

Workflow files under `src/biomodals/workflow/` orchestrate multiple Biomodals apps into a longer-running Modal pipeline. Keep them thin, explicit, and compatible with the app wrappers they call.

## File and Interface Conventions

- Name workflow files `<workflow-name>_workflow.py`.
- Define a module-level `CONF = AppConfig(...)` and `app = modal.App(...)`, as app files do.
- Keep module docstrings user-facing. Include upstream workflow links, required deployed apps, persistent output behavior, and example commands.
- Prefer upstream-compatible config files when wrapping an upstream scheduler. Avoid inventing a new schema unless the workflow needs one.
- Local entrypoints should be named `submit_<workflow-name>_workflow(...)` and use Google-style `Args:` docstrings. Workflow files are currently invoked directly with `modal run`; check their flags with `modal run src/biomodals/workflow/<name>_workflow.py --help`.

## Deployed App Calls

- Invoke deployed apps with `modal.Function.from_name(app_name, function_name)`.
- Keep deployed app names configurable with environment variables when workflows depend on independently deployed apps.
- Treat app wrappers as public interfaces. Make backward-compatible app changes when a workflow needs better path control, returned artifacts, or volume staging.
- Document every app-interface adaptation in `docs/agents/`, including why the workflow could not use the existing function directly.

## Queues and Parallelism

- Use `modal.Queue` for active, per-run work distribution only. Do not rely on queues as durable state.
- Give queues run-scoped names that include a sanitized run id, and delete named queues after workers finish.
- Parallelize independent work at the smallest useful boundary, usually per structure, per batch, or per scoring pair.
- Use an existing app's internal batching or queueing when it already provides the needed concurrency.
- Return small structured results from workers and store large artifacts in Modal volumes or returned tarballs.

## Output and Volume Layout

- Preserve upstream output layouts when users are likely to compare results against upstream docs or scripts.
- Use deterministic stage directories such as `stage1/`, `stage2/`, and `design_output/` for multi-step workflows.
- Keep intermediate artifacts that are needed for resuming, filtering, ranking, or audit logs. Avoid keeping raw cache directories that are only useful inside one worker.
- Package final outputs with `package_outputs(...)` as `.tar.zst` bytes when direct local download is practical.
- Avoid overwriting existing local archives unless the entrypoint exposes an explicit `force` option.

## Validation

- During development, small local unit tests or scratch scripts are fine, but do not commit generated test files unless the user asks for them.
- Run `uv run biomodals list` and `uv run biomodals help <app-name>` when app discovery or app docstrings change. For workflows, run `uv run modal run src/biomodals/workflow/<name>_workflow.py --help`.
- Run `prek run --files <changed files>` when practical before committing.
- Do not claim that a workflow works end-to-end on Modal unless it has actually been run on Modal.
