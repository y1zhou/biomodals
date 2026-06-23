# Run expensive PPIFlow stage coordinators remotely

Expensive PPIFlow stage coordinators run as remote workflow nodes, even when their implementation mostly submits child app calls. This keeps one recoverable Modal call identity per static workflow stage, lets the stage write a durable candidate manifest, and allows retries to skip completed candidates instead of redoing or losing mixed-success work inside the orchestrator.
