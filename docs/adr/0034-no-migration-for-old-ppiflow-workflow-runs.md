# No migration for old PPIFlow workflow runs

The PPIFlow workflow refactor does not migrate old in-progress workflow ledgers or artifact manifests. PPIFlow was already marked incomplete as a reference pattern, and the refactor changes candidate identity and artifact contracts; old in-progress runs should be restarted with `force`, while useful completed app-owned outputs can be reintroduced through explicit `Stage2Input`.
