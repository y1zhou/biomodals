# biomodals

Bioinformatics tools running on modal.
Note that this repository is heavily refactored from [the upstream repository](https://github.com/hgbrian/biomodals).
All new apps have the `_app.py` suffix to distinguish from the original ones.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/y1zhou/biomodals)

## Installation

```bash
git clone https://github.com/y1zhou/biomodals.git
cd biomodals
pip install .
biomodals --help
```

Or alternatively, use [uv](https://github.com/astral-sh/uv), e.g.:

```bash
git clone https://github.com/y1zhou/biomodals.git
cd biomodals
uv run biomodals --help
```

## Getting started

To see a list of all available commands, run:

```bash
biomodals --help
```

To list and inspect apps:

```bash
biomodals app list
biomodals app help <app-name>
```

To list and inspect workflows:

```bash
biomodals workflow list
biomodals workflow help <workflow-name>
```

To run a workflow, pass workflow-specific flags after `--`:

```bash
uv run biomodals workflow run ppiflow --dry-run -- \
  --task-yaml examples/data/ppiflow_workflow_task.yaml \
  --steps-yaml examples/data/ppiflow_workflow_steps.yaml
```

## API server

Start the GROMACS API as a hot-reloading Modal development endpoint:

```bash
uv run modal serve -m biomodals.app.service.gromacs_api_app
```

Modal prints the temporary API URL. Requests require proxy-token credentials,
which you can create with:

```bash
uv run modal workspace proxy-tokens create
```
