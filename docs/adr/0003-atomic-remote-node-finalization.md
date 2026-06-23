# Finalize remote node success as one durable operation

A remote call will remain recoverable until its processed app result, materialized files, artifact manifests, attempt state, node state, and remote-call status are recorded under one workflow-volume synchronization lock and committed together. This ordering prevents a reload from discarding uncommitted ledger mutations and prevents preemption from leaving a successful call without the result needed for recovery.
