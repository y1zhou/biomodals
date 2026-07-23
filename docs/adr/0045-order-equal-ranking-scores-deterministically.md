# Order equal ranking scores deterministically

Status: accepted.

Request-scoped and global prediction rankings use the same total order:

1. descending `ranking_score`;
2. ascending model seed;
3. ascending sample index.

The first row is the best prediction used for the corresponding top-level
files. The seed and sample tie-breakers preserve upstream's first-result-wins
semantics after seed normalization while removing dependence on container
partitioning, completion order, and summary rebuild timing.

Equal-score samples remain scientifically equivalent. The tie-breaker only
stabilizes presentation and artifact selection; it does not claim a scientific
preference between them.
