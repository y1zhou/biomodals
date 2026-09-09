# Documentation

## Using BioModals

- [Getting started and CLI/API operations](../README.md)
- [Humanizing antibodies](humanization.md)
- [Production and pre-release deployment](deployment/mvp-runbook.md)

## Developing BioModals

Read [repository instructions](../AGENTS.md) before making changes. These roles
distinguish current behavior from the reasons and procedures behind it.

| Question | Source |
| --- | --- |
| What does a domain term mean? | [Domain glossary](../CONTEXT.md) |
| What behavior must an implementation preserve? | [Specifications](specs/) |
| Why was an architectural choice made? | [Decision records](adr/) |
| How should an agent change an app or workflow? | [App skill](../.agents/skills/biomodals-app-development/SKILL.md), [workflow skill](../.agents/skills/biomodals-workflow-development/SKILL.md) |
| Which exceptions are approved? | [App coordination](agents/app-development.md) |
| What source evidence supports a scientific claim? | [Research](research/) |

Humanization has one [scientific/workflow contract](specs/humanization-workflow.md)
and one [website/API contract](specs/humanization-service.md). Its
[publication decision](adr/0010-humanization-publication-contract.md) explains
the output boundary without restating the field contract.

Shared execution behavior is defined by the [scheduler spec](specs/unified-task-scheduler.md)
and its [consolidation amendment](specs/execution-kernel-consolidation.md).
The [API service contract](specs/api-tool-service.md) and
[remote-authority decision](adr/0007-api-jobs-use-remote-coordinators.md)
define the service/coordinator boundary. Follow their explicit supersession
notices when consulting older ADRs; history is not an alternative implementation.

Research snapshots preserve cited evidence, not release instructions. Current
defaults, CLI arguments and test commands come from source, `--help`, and the
repository checks rather than historical smoke-test reports.
