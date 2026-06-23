# Treat artifact availability as available, missing, or unknown

Artifact verification will treat an explicit missing result as authoritative, checker failures and unmounted volumes as unknown, and only missing artifacts as grounds to invalidate completed node state. A producer rerun must verify its new outputs before durable completion; outputs that remain missing fail the attempt rather than entering an unbounded retry loop.
