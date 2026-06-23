# Keep PPIFlow Rosetta job coordination in the workflow

PPIFlow owns Rosetta job manifests, script and flags selection, queue setup, queue cleanup, expected outputs, and per-candidate Rosetta status. The generic Rosetta app remains a command worker so PPIFlow-specific interface-energy and relax semantics do not leak into the shared Rosetta app API before they prove reusable.
