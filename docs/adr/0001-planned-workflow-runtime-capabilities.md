# Superseded: planned workflow runtime capabilities

Status: superseded by [ADR 0006](0006-unified-execution-kernel.md).

This document described the pre-kernel `WorkflowLedger`, remote-call manager,
attempt, and generic rerun-policy design. Those interfaces and schema were
deleted during the direct workflow cutover and must not be used as
implementation guidance.

The retained decisions now have these forms:

- semantic workflow DAGs remain fixed, while runtime-discovered cardinality is
  represented by Tasks inside a Node;
- Provider Calls are preclaimed and durably attached through
  `biomodals.execution`; uncertain ownership never authorizes replacement work;
- Tasks have no attempt identity and receive one scheduler submission per
  Execution Run;
- `resume` continues a suspended Run or explicitly reconciles
  `state_unknown`, and never retries conclusive failure;
- retrying missing work requires an explicit compatible Successor Execution Run;
- workflow-owned caches and scientific publications remain outside the kernel
  and are validated with `available`, `missing`, or `unknown` observations;
- mutable DAGs, a shared scientific cache schema, and blind remote-call
  replacement remain rejected.

See ADR 0006, the
[unified scheduler specification](../specs/unified-task-scheduler.md), and the
workflow development guide for the current contracts.
