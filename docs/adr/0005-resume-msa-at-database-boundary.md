# Resume MSA search at the database-result boundary

Status: accepted.

The durable retry and cache boundary for sharded MSA search is one Raw Database
MSA: the validated result for one unique sequence against one reference
database profile. The coordinator submits one Modal worker invocation for each
missing sequence-by-database result, while that worker owns the database's
internal shard fanout. It reuses independently validated database results and
constructs AlphaFold's combined MSAs in the pinned upstream order only after
all required results are available.

This boundary preserves successful database searches across a preemption and
matches the resource topology measured by the sharding benchmarks without
introducing per-shard durable scheduling, publication, and repair state. A
worker preempted partway through its database search must rerun that whole
database; making individual shards durable remains a future optimization if
measurements show that this retry cost justifies the extra state.
