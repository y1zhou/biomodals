# Use stage-specific PPIFlow remote wrappers

PPIFlow uses stage-specific Modal remote wrapper functions around shared candidate-wide coordinator helpers. Shared logic remains in `ppiflow/coordinators.py`, while separate wrapper names make logs, failures, mounts, and future resource settings easier to understand and tune per stage.
