# Validate canonical run names at reusable app boundaries

Local entrypoints may normalize a human-provided run name once and report the canonical value, but remote and workflow-compatible app functions will reject non-canonical names. App Run Layout construction for mounted volumes will validate containment so traversal is impossible and distinct user inputs cannot silently collapse onto the same durable cache directory.
