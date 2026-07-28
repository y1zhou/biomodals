# GROMACS analysis checkpoint exception

`collect_traj_stats` predates the staged-cache manifest standard and is shared
by the established GROMACS CLI entrypoint and API workflow. Within one fixed
run directory, it treats the RMSD, radius-of-gyration, and RMSF CSV/PNG members
as restart checkpoints and uses their modification times relative to the input
trajectory to identify stale members.

For the pre-release MVP, this behavior is intentionally retained so the API can
repair a missing plot without changing the standalone CLI contract or
recomputing completed molecular dynamics. The exception is narrow:

- it does not reuse outputs across run names or Inputs;
- a missing or stale member is regenerated and the pair is committed together;
- the API structurally validates every required CSV and PNG before publishing a
  successful Result; and
- no new GROMACS cache stage may copy this timestamp-only contract.

A future change that shares analysis outputs across identities, changes their
scientific policy, or adds another reusable stage must replace this exception
with a versioned manifest containing the trajectory identity, analysis policy,
expected members, and validated artifact facts described by the app-development
staged-cache standard.
