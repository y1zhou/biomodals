"""ABCFold2 Modal app client code.

Example usage:

    python -i clients/fold/abcfold2.py
    > run_abcfold2("path/to/example.yaml")
"""

import modal

download_boltz_models = modal.Function.from_name("ABCFold2", "download_boltz_models")
download_chai_models = modal.Function.from_name("ABCFold2", "download_chai_models")
prepare_abcfold2 = modal.Function.from_name("ABCFold2", "prepare_abcfold2")
collect_abcfold2_boltz_data = modal.Function.from_name(
    "ABCFold2", "collect_abcfold2_boltz_data"
)
collect_abcfold2_chai_data = modal.Function.from_name(
    "ABCFold2", "collect_abcfold2_chai_data"
)


# from biomodals.app.fold.abcfold2 import submit_abcfold2_task
# This doesn't work directly because the function references undeployed modal functions.
# It would raise an ExecutionError:
# Function has not been hydrated with the metadata it needs to run on Modal, because the
# App it is defined on is not running.
def submit_abcfold2_task(
    input_yaml: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    download_models: bool = False,
    force_redownload: bool = False,
    run_boltz: bool = True,
    run_chai: bool = True,
    # TODO: modify GPU types and timeouts as needed
) -> None:
    """Run ABCFold2 on modal and fetch results to `out_dir`.

    Args:
        input_yaml: Path to YAML design specification file
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to {input filename stem})
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
        run_boltz: Whether to run Boltz inference
        run_chai: Whether to run Chai inference
    """
    import json
    from pathlib import Path

    # Load input and find its hash
    yaml_path = Path(input_yaml).expanduser().resolve()
    yaml_str = yaml_path.read_bytes()

    if run_name is None:
        run_name = yaml_path.stem

    if out_dir is None:
        out_dir = Path.cwd()
    local_out_dir = Path(out_dir) / run_name
    if local_out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {local_out_dir}")

    print("🧬 Starting ABCFold2 run...")
    run_conf = prepare_abcfold2.remote(yaml_str=yaml_str)
    local_out_dir.mkdir(parents=True, exist_ok=True)
    with open(local_out_dir / "run-config.json", "w") as f:
        json.dump(run_conf, f, indent=2)

    if download_models:
        print("🧬 Checking Boltz inference dependencies...")
        download_boltz_models.remote(force=force_redownload)

        print("🧬 Checking Chai inference dependencies...")
        download_chai_models.remote(force=force_redownload)

    # Run Boltz for each seed
    if run_boltz:
        out_path = local_out_dir / "boltz_models.tar.zst"
        print(f"🧬 Running Boltz and collecting results to {out_path}")
        boltz_data = collect_abcfold2_boltz_data.remote(run_conf=run_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(boltz_data)

    # Run Chai for each seed
    if run_chai:
        out_path = local_out_dir / "chai_models.tar.zst"
        print(f"🧬 Running Chai and collecting results to {out_path}")
        chai_data = collect_abcfold2_chai_data.remote(run_conf=run_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(chai_data)

    print(f"🧬 ABCFold2 run complete! Results saved to {local_out_dir}")
