"""Materialize the declared RNA image source mounts for old-interpreter CI.

This uses Modal's local image dependency graph; it performs no provider calls.
The resulting source trees intentionally exclude unrelated checkout modules.
"""

import argparse
import importlib
import shutil
from pathlib import Path

import orjson

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
report = {}
for name in ("oligoformer", "ensirna"):
    module = importlib.import_module(f"biomodals.app.score.{name}_app")
    destination = args.output / name
    pending, seen, files = [module.runtime_image], set(), {}
    while pending:
        dependency = pending.pop()
        if id(dependency) in seen:
            continue
        seen.add(id(dependency))
        pending.extend(dependency._deps_())
        for entry in getattr(dependency, "entries", ()):
            for local, remote in entry.get_files_to_upload():
                if str(remote).startswith("/root/biomodals/"):
                    target = destination / Path(remote).relative_to("/root")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, target)
                    files[str(remote)] = str(local)
    report[name] = files
(args.output / "mounts.json").write_bytes(
    orjson.dumps(report, option=orjson.OPT_INDENT_2)
)
