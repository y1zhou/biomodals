"""ABCFold2 Modal app client code.

Example usage:

    python -i clients/fold/abcfold2.py
    > run_abcfold2("path/to/example.yaml")
"""

import modal

from biomodals.app.fold.abcfold2 import run_abcfold2  # noqa: F401

download_boltz_models = modal.Function.from_name("ABCFold2", "download_boltz_models")
download_chai_models = modal.Function.from_name("ABCFold2", "download_chai_models")
prepare_abcfold2 = modal.Function.from_name("ABCFold2", "prepare_abcfold2")
collect_abcfold2_boltz_data = modal.Function.from_name(
    "ABCFold2", "collect_abcfold2_boltz_data"
)
collect_abcfold2_chai_data = modal.Function.from_name(
    "ABCFold2", "collect_abcfold2_chai_data"
)

if __name__ == "__main__":
    print("Use 'run_abcfold2(\"path/to/example.yaml\")' to submit tasks.")
