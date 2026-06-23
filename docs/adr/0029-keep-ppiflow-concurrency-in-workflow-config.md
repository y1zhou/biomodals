# Keep PPIFlow concurrency in workflow config

Shared PPIFlow candidate concurrency lives in the existing task/steps YAML parsing layer and is copied into node configuration during DAG construction. It does not become a workflow runtime or orchestrator flag because candidate fan-out is a PPIFlow workflow concern, not a general scheduling API.
