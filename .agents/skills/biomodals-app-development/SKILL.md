---
name: biomodals-app-development
description: Biomodals Modal app development standards. Use when Codex is creating, editing, reviewing, or scaffolding files under src/biomodals/app/**/*_app.py or their supporting modules, including discovery, AppConfig and image construction, input safety, execution-kernel integration, remote coordinators, staged caches, fanout and resumption, upstream patches, scientific equivalence, workflow outputs, examples, and tests.
---

# Biomodals App Development

## Workflow

1. Read [repo coordination](../../../docs/agents/app-development.md) for approved
   deviations.
2. Read `references/quick-app.md` for every app change.
3. Load only what the app needs:
   - [Python runtime compatibility](../../../docs/agents/python-runtime-compatibility.md):
     any Modal image runs below the repository's Python minimum or imports
     Biomodals source built for a different Python version.
   - `references/multi-module.md`: multiple stages/backends obscure composition.
   - `references/execution-kernel.md`: durable multi-call scheduling, remote
     coordinators, restart, or service/workflow-owned child calls.
   - `references/staged-cache.md`: persistent or resumable artifacts.
   - `references/fanout-resume.md`: parallel or interruptible work; also load
     `execution-kernel.md` for durable scheduling and `staged-cache.md` for
     reusable Volume publications.
   - `references/upstream-patching.md`: upstream execution, patches, or
     equivalence claims.
   - `references/testing.md`: behavior, invocation, cache, or orchestration
     changes.
4. If the app must compose into workflows, also use
   `biomodals-workflow-development`.

## Across all app shapes

- Preserve `*_app.py` discovery and help behavior; complex apps may delegate to
  sibling modules.
- Give generic graph, artifact, Run, Node, Task, Provider Call, coordinator, and
  restart mechanics to `biomodals.execution`; use
  `biomodals.execution.modal` for Modal hosting. Keep scientific plans, cache
  probes, inputs, result decoding, and publications app-owned.
- Build the same `ExecutionDefinition` whether an app runs from the CLI, a
  workflow, or the API service. Add workload behavior through Execution Nodes,
  not app-local lifecycle or admission loops.
- Treat user-controlled content as untrusted across paths and processes.
- Version scientific deviations into affected cache identities.
- Publish validated artifacts before completion markers and cleanup.
- Add focused tests, examples, and the repository checks in `testing.md`.
