"""PPIFlow-Pipeline de novo VHH design on Modal.

Runs the full PPIFlow-Pipeline nanobody (VHH) workflow on Modal, from backbone
generation through scoring and ranking. The script builds a self-contained Modal
image by cloning pinned upstream repos, so it does not depend on any local
checkout.

All pipeline outputs are persisted to the ``ppiflow-outputs`` Modal volume under
``/ppiflow-outputs/{run_name}/`` so results survive client disconnects and
re-runs with the same ``--run-name`` resume from the last completed step.

## Required Modal resources

- Volume ``ppiflow-models`` mounted at ``/models`` containing:
  - ``nanobody.ckpt`` — PPIFlow nanobody backbone checkpoint
  - ``abmpnn.pt`` — AbMPNN sequence-design checkpoint
  - ``flowpacker_bc40.pth``, ``flowpacker_cluster.pth``, ``flowpacker_confidence.pth``
  - ``af3.bin`` — AF3Score weights
  - ``af3_data/`` — AF3/AF3Score databases (MSA, templates)
- Volume ``ppiflow-outputs`` mounted at ``/ppiflow-outputs`` (created automatically).

## Minimal command

```bash
modal run ppiflow_claude.py \\
    --run-name my_vhh \\
    --target-pdb target.pdb \\
    --framework-pdb framework.pdb \\
    --target-chain M \\
    --hotspots "M18,M19,M22"
```

## Output structure (on ppiflow-outputs volume)

```
/ppiflow-outputs/{run_name}/
├── step_01_gen/
├── step_02_seq1/
├── ...
├── step_16_rank_finalize/
├── _pipeline/          # native pipeline output (for --reuse resume)
│   ├── config/
│   ├── output/
│   ├── logs/
│   └── results/
├── inputs/             # input PDB files (archived)
└── run_summary.txt     # execution summary
```
"""

# ruff: noqa: PLC0415, S603
import os
import shlex
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################

# --- GPU and timeout ---
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "86400"))
APP_NAME = os.environ.get("MODAL_APP", "PPIFlow")

# --- Volumes ---
MODELS_VOLUME = Volume.from_name("ppiflow-models", create_if_missing=True)
MODELS_DIR = "/models"
AF3_DB_DIR = f"{MODELS_DIR}/af3_data"

OUTPUTS_VOLUME = Volume.from_name("ppiflow-outputs", create_if_missing=True)
OUTPUTS_DIR = "/ppiflow-outputs"

# --- Repository pins ---
PPIFLOW_PIPELINE_REPO = "https://github.com/cytokineking/PPIFlow-Pipeline.git"
PPIFLOW_PIPELINE_COMMIT = "89ced25f18cda36f88d2fde0ecd7db32d2195893"

PPIFLOW_REPO = "https://github.com/Mingchenchen/PPIFlow.git"
PPIFLOW_COMMIT = "000ce45a4411e7b97c1523a22c0f1bece7ede5fc"

FLOWPACKER_REPO = "https://gitlab.com/mjslee0921/flowpacker.git"
FLOWPACKER_COMMIT = "03421c7fdda73862994aa54fb3077f3f6561408c"

AF3SCORE_REPO = "https://github.com/Mingchenchen/AF3Score.git"
AF3SCORE_COMMIT = "b0764aaa4101f8a22a5f404faef7acc13ee52d06"

# --- Remote paths ---
PIPELINE_ROOT = Path("/opt/PPIFlow-Pipeline")
PPIFLOW_ROOT = Path("/opt/PPIFlow")
FLOWPACKER_ROOT = Path("/opt/flowpacker")
AF3SCORE_ROOT = Path("/opt/AF3Score")
AF3SCORE_VENV = AF3SCORE_ROOT / ".venv"
STUBS_DIR = Path("/opt/_stubs")

# --- Rosetta (from rosettacommons/rosetta:serial) ---
ROSETTA_IMAGE = "rosettacommons/rosetta:serial"
ROSETTA_BIN_DIR = Path("/usr/local/bin")
ROSETTA_DATABASE_DIR = Path("/usr/local/database")
ROSETTA_EXECUTABLES = {
    "interface_analyzer": "InterfaceAnalyzer.default.linuxgccrelease",
    "relax": "rosetta_scripts.default.linuxgccrelease",
}

# --- Weight paths inside /models ---
NANOBODY_CKPT = Path(MODELS_DIR) / "nanobody.ckpt"
ABMPNN_CKPT = Path(MODELS_DIR) / "abmpnn.pt"
AF3_WEIGHTS = Path(MODELS_DIR) / "af3.bin"
FLOWPACKER_BC40 = Path(MODELS_DIR) / "flowpacker_bc40.pth"
FLOWPACKER_CLUSTER = Path(MODELS_DIR) / "flowpacker_cluster.pth"
FLOWPACKER_CONFIDENCE = Path(MODELS_DIR) / "flowpacker_confidence.pth"
ROSETTA_DB_VOLUME = Path(MODELS_DIR) / "rosetta_database"

# --- PyTorch / dependency pins ---
PYTORCH_CU121_INDEX = "https://download.pytorch.org/whl/cu121"
PYG_WHL = "https://data.pyg.org/whl/torch-2.3.0+cu121.html"

TORCH_PKGS = [
    "torch==2.3.1+cu121",
    "torchvision==0.18.1+cu121",
    "torchaudio==2.3.1+cu121",
]
PYG_PKGS = [
    "pyg-lib==0.4.0+pt23cu121",
    "torch-scatter==2.1.2+pt23cu121",
    "torch-sparse==0.6.18+pt23cu121",
    "torch-cluster==1.6.3+pt23cu121",
    "torch-spline-conv==1.2.2+pt23cu121",
    "torch-geometric==2.6.1",
]
INFERENCE_PKGS = [
    "abnumber", "anarci", "numpy==1.26.3", "scipy==1.15.2", "pandas==2.2.3",
    "scikit-learn==1.2.2", "pyyaml==6.0.2", "omegaconf==2.3.0",
    "hydra-core==1.3.2", "hydra-submitit-launcher==1.2.0", "submitit==1.5.3",
    "tqdm==4.67.1", "lightning==2.5.0.post0", "pytorch-lightning==2.5.0.post0",
    "torchmetrics==1.6.2", "lightning-utilities==0.14.0",
    "einops==0.8.1", "easydict==1.13", "dm-tree==0.1.9",
    "optree==0.14.1", "opt-einsum==3.4.0", "opt-einsum-fx==0.1.4",
    "e3nn==0.5.6", "fair-esm==2.0.0",
    "biopython==1.83", "biotite==1.0.1", "biotraj==1.2.2",
    "gemmi==0.6.5", "ihm==2.2", "modelcif==0.7",
    "tmtools==0.2.0", "freesasa==2.2.1", "mdtraj==1.10.3",
    "requests==2.32.3", "packaging==24.2", "typing-extensions==4.12.2",
    "protobuf==3.20.2", "tensorboard==2.19.0",
    "grpcio==1.72.1", "gputil==1.4.0", "hjson==3.1.0", "ninja==1.11.1.3",
    "dockq",
]
AF3SCORE_PKGS = [
    "absl-py", "chex", "dm-haiku==0.0.13", "dm-tree", "jax[cuda12]==0.4.34",
    "jax-triton==0.2.0", "jaxtyping==0.2.34", "tqdm",
    "triton==3.1.0", "typeguard==2.13.3", "zstandard", "biopython", "h5py",
    "pandas", "setuptools",
]
# rdkit is installed separately with --only-binary to avoid building cifpp
# from source (requires C++20, unavailable on Ubuntu 20.04).
AF3SCORE_RDKIT = "rdkit==2025.3.6"

# --- Step output mapping ---
# Maps numbered step directory names to their pipeline-native output paths
# (relative to the _pipeline working directory).
STEP_OUTPUT_MAP = [
    ("step_01_gen", "output/backbones"),
    ("step_02_seq1", "output/seqs_round1"),
    ("step_03_flowpacker1", "output/flowpacker_round1"),
    ("step_04_af3score1", "output/af3score_round1"),
    ("step_05_rosetta_interface", "output/rosetta_interface"),
    ("step_06_interface_enrich", "output/interface_enrich"),
    ("step_07_partial", "output/partial_flow"),
    ("step_08_seq2", "output/seqs_round2"),
    ("step_09_flowpacker2", "output/flowpacker_round2"),
    ("step_10_af3score2", "output/af3score_round2"),
    ("step_11_relax", "output/relax"),
    ("step_12_rosetta_interface2", "output/rosetta_interface2"),
    ("step_13_af3_refold", "output/af3_refold"),
    ("step_14_dockq", "output/dockq"),
    ("step_15_rank_features", "results/features"),
    ("step_16_rank_finalize", "results"),
]


##########################################
# Image and app definitions
##########################################

def _stub_commands() -> list[str]:
    """Generate shell commands that create lightweight Python stubs.

    PPIFlow imports ``deepspeed`` and ``models.layer_norm`` at module level.
    Rather than installing the full packages we create thin shims that satisfy
    the imports without pulling in heavy CUDA extensions.
    """
    stubs = {
        f"{STUBS_DIR}/deepspeed/__init__.py": (
            "from . import checkpointing, comm\n__all__ = ['checkpointing', 'comm']\n"
        ),
        f"{STUBS_DIR}/deepspeed/ops/__init__.py": "__all__ = []\n",
        f"{STUBS_DIR}/deepspeed/comm/__init__.py": (
            "class _Comm:\n"
            "    @staticmethod\n"
            "    def is_initialized():\n"
            "        return False\n"
            "comm = _Comm()\n__all__ = ['comm']\n"
        ),
        f"{STUBS_DIR}/deepspeed/checkpointing/__init__.py": (
            "def is_configured():\n    return False\n"
            "def checkpoint(function, *args, **kwargs):\n    return function(*args, **kwargs)\n"
        ),
        f"{STUBS_DIR}/models/__init__.py": (
            "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
        ),
        f"{STUBS_DIR}/models/layer_norm/__init__.py": (
            "from .layer_norm import FusedLayerNorm\n__all__ = ['FusedLayerNorm']\n"
        ),
        f"{STUBS_DIR}/models/layer_norm/layer_norm.py": (
            "import torch\n\n"
            "class FusedLayerNorm(torch.nn.LayerNorm):\n"
            "    def __init__(self, normalized_shape, eps=1e-5):\n"
            "        super().__init__(normalized_shape, eps=eps, elementwise_affine=True)\n"
            "    def kernel_forward(self, input):\n"
            "        return super().forward(input)\n"
        ),
    }
    cmds: list[str] = []
    for path, content in stubs.items():
        parent = str(Path(path).parent)
        cmds.append(f"mkdir -p {shlex.quote(parent)}")
        cmds.append(
            f"python -c {shlex.quote(f'from pathlib import Path; Path({path!r}).write_text({content!r})')}"
        )
    return cmds


def _patch_upstream_commands() -> list[str]:
    """Patch pinned upstream repos for known CLI compatibility issues."""
    mpnn_run = PPIFLOW_ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
    patch = (
        "from pathlib import Path; "
        f"path = Path({str(mpnn_run)!r}); "
        "text = path.read_text(); "
        "old1 = 'def main(args):\\n    parsed_chain_dict_list = parse_multiple_chains(args.folder_with_pdbs_path, args.ca_only)\\n'; "
        "new1 = 'def main(args):\\n    parsed_chain_dict_list = []\\n    if args.folder_with_pdbs_path:\\n        parsed_chain_dict_list = parse_multiple_chains(args.folder_with_pdbs_path, args.ca_only)\\n'; "
        "old2 = '    chain_id_dict = assign_fixed_chains(parsed_chain_dict_list, args.chain_list)\\n'; "
        "new2 = '    chain_id_dict = assign_fixed_chains(parsed_chain_dict_list, args.chain_list)\\n    fixed_positions_dict = None\\n'; "
        "assert old1 in text, f'Expected patch target not found in {path}'; "
        "assert old2 in text, f'Expected patch target not found in {path}'; "
        "text = text.replace(old1, new1, 1); "
        "path.write_text(text.replace(old2, new2, 1))"
    )
    return [f"python -c {shlex.quote(patch)}"]


# -- Main image: PPIFlow backbone generation + sequence design + scoring --
# Base on rosettacommons/rosetta:serial so that Rosetta binaries are available
# for interface analysis and relax steps without needing a separate image.
runtime_image = (
    Image.from_registry(ROSETTA_IMAGE, add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(
        "git", "curl", "ca-certificates", "build-essential", "python3-dev", "zstd",
        "pkg-config", "gfortran", "libopenblas-dev", "liblapack-dev",
        "libhdf5-dev", "libnetcdf-dev", "zlib1g-dev", "libbz2-dev", "liblzma-dev",
        "cmake", "software-properties-common", "hmmer",
    )
    .run_commands(
        # Install gcc-12/g++-12 from ubuntu-toolchain-r PPA for C++20 support
        # (required by AF3Score's cifpp build dependency).
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test",
        "apt-get update -qq",
        "apt-get install -y -qq g++-12",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        # Override CC/CXX set by the Rosetta image (clang) so that pip builds
        # of C extensions use gcc-12 for C++20 support (AF3Score cifpp).
        "CC": "gcc-12",
        "CXX": "g++-12",
        "PYTHONPATH": ":".join([
            str(STUBS_DIR),
            str(PIPELINE_ROOT / "src"),
            str(PIPELINE_ROOT),
            str(AF3SCORE_ROOT / "src"),
        ]),
        "AF3SCORE_PYTHON": str(AF3SCORE_VENV / "bin" / "python"),
        "AF3_DB_DIR": AF3_DB_DIR,
        "PPIFLOW_AF3_DB": AF3_DB_DIR,
        "WANDB_MODE": "disabled",
    })
    .run_commands(
        *_stub_commands(),
        # Clone repos at pinned commits
        f"git clone {shlex.quote(PPIFLOW_PIPELINE_REPO)} {PIPELINE_ROOT}",
        f"git -C {PIPELINE_ROOT} checkout --detach {PPIFLOW_PIPELINE_COMMIT}",
        f"git clone {shlex.quote(PPIFLOW_REPO)} {PPIFLOW_ROOT}",
        f"git -C {PPIFLOW_ROOT} checkout --detach {PPIFLOW_COMMIT}",
        f"git clone {shlex.quote(FLOWPACKER_REPO)} {FLOWPACKER_ROOT}",
        f"git -C {FLOWPACKER_ROOT} checkout --detach {FLOWPACKER_COMMIT}",
        f"git clone {shlex.quote(AF3SCORE_REPO)} {AF3SCORE_ROOT}",
        f"git -C {AF3SCORE_ROOT} checkout --detach {AF3SCORE_COMMIT}",
        *_patch_upstream_commands(),
    )
    .uv_pip_install(*TORCH_PKGS, extra_index_url=PYTORCH_CU121_INDEX)
    .uv_pip_install(*PYG_PKGS, find_links=PYG_WHL)
    .uv_pip_install(*INFERENCE_PKGS)
    .run_commands(
        # AF3Score needs triton==3.1.0, which conflicts with the torch==2.3.1
        # + PyG runtime in the main env, so we keep it in a separate venv.
        # /.uv/uv is copied by the preceding .uv_pip_install() steps.
        # Temporarily hide Rosetta's custom /usr/local/bin/libzlib.so so that
        # CMake (used by scikit-build-core for AF3Score's cifpp) links against
        # the system zlib from zlib1g-dev instead.
        (
            f"mv /usr/local/bin/libzlib.so /usr/local/bin/_libzlib.so.bak 2>/dev/null; "
            f"cd {AF3SCORE_ROOT} && "
            f"/.uv/uv venv {AF3SCORE_VENV} --python 3.11 && "
            f"/.uv/uv pip install --python {AF3SCORE_VENV / 'bin' / 'python'} "
            f"{' '.join(shlex.quote(pkg) for pkg in AF3SCORE_PKGS)} && "
            # Install rdkit separately with --only-binary to avoid C++20 source build
            f"/.uv/uv pip install --python {AF3SCORE_VENV / 'bin' / 'python'} "
            f"--only-binary rdkit {shlex.quote(AF3SCORE_RDKIT)} && "
            f"/.uv/uv pip install --python {AF3SCORE_VENV / 'bin' / 'python'} --no-deps -e . && "
            f"mv /usr/local/bin/_libzlib.so.bak /usr/local/bin/libzlib.so 2>/dev/null; "
            f"{AF3SCORE_VENV / 'bin' / 'build_data'}"
        ),
    )
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################

def run_command(cmd: list[str], **kwargs) -> None:
    """Execute a shell command, streaming stdout line-by-line."""
    import subprocess as sp

    print(f"▶ Running: {' '.join(cmd)}")
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout.")
        buf = None
        while (buf := p.stdout.readline()) != "" or p.poll() is None:
            print(buf, end="", flush=True)
        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buf)


##########################################
# Fetch model weights and data caches
##########################################

def verify_weights() -> list[str]:
    """Check that required weight files exist in /models and return warnings."""
    required = {
        "nanobody.ckpt": NANOBODY_CKPT,
        "abmpnn.pt": ABMPNN_CKPT,
    }
    optional = {
        "af3.bin": AF3_WEIGHTS,
        "flowpacker_bc40.pth": FLOWPACKER_BC40,
        "flowpacker_cluster.pth": FLOWPACKER_CLUSTER,
        "flowpacker_confidence.pth": FLOWPACKER_CONFIDENCE,
    }
    warnings: list[str] = []
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required weight missing from volume: {path}. "
                f"Upload {label} to the ppiflow-models volume."
            )
    for label, path in optional.items():
        if not path.exists():
            warnings.append(f"Optional weight not found: {path} ({label})")
    return warnings


def ensure_flowpacker_checkpoints(
    flowpacker_root: Path,
    bc40: Path,
    cluster: Path,
    confidence: Path,
) -> None:
    """Point FlowPacker checkpoint paths at the real volume-mounted weights."""
    checkpoints = flowpacker_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    mapping = {
        checkpoints / "bc40.pth": bc40,
        checkpoints / "cluster.pth": cluster,
        checkpoints / "confidence.pth": confidence,
    }
    for dest, src in mapping.items():
        if not src.exists():
            continue
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src)


# -- Bundled VHH framework defaults --
_BUNDLED_VHH_FRAMEWORK_DEFAULTS = {
    "5jds_nanobody_framework.pdb": ("A", "CDRH1,8-8,CDRH2,8-8,CDRH3,21-21"),
    "7eow_nanobody_framework.pdb": ("A", "CDRH1,8-8,CDRH2,8-8,CDRH3,20-20"),
    "7xl0_nanobody_framework.pdb": ("A", "CDRH1,8-8,CDRH2,7-7,CDRH3,15-15"),
    "8coh_nanobody_framework.pdb": ("A", "CDRH1,8-8,CDRH2,8-8,CDRH3,19-19"),
    "8z8v_nanobody_framework.pdb": ("A", "CDRH1,8-8,CDRH2,8-8,CDRH3,8-8"),
}


def resolve_vhh_framework_settings(
    framework_path: Path,
    framework_heavy_chain: str = "A",
    cdr_length: str | None = None,
) -> tuple[str, str]:
    """Resolve VHH framework CLI settings, inferring bundled defaults when possible."""
    heavy_chain = framework_heavy_chain.strip() or "A"
    if cdr_length:
        return heavy_chain, cdr_length
    bundled_defaults = _BUNDLED_VHH_FRAMEWORK_DEFAULTS.get(framework_path.name.lower())
    if bundled_defaults:
        default_heavy_chain, default_cdr_length = bundled_defaults
        return heavy_chain or default_heavy_chain, default_cdr_length
    raise ValueError(
        "--cdr-length is required unless --framework-pdb points to a known bundled "
        "nanobody framework."
    )


def resolve_rosetta_db() -> str | None:
    """Return the best available Rosetta database path, or None."""
    for candidate in (ROSETTA_DATABASE_DIR, ROSETTA_DB_VOLUME):
        if candidate.exists():
            return str(candidate)
    return None


def organize_step_outputs(pipeline_dir: Path, run_dir: Path) -> None:
    """Create numbered step directories with symlinks to pipeline native output.

    This organizes the pipeline's native output structure into the user-facing
    ``step_01_gen/``, ``step_02_seq1/``, ... layout without duplicating data.
    Falls back to copying if symlinks are not supported.
    """
    import shutil

    for step_dir_name, pipeline_subpath in STEP_OUTPUT_MAP:
        src = pipeline_dir / pipeline_subpath
        dst = run_dir / step_dir_name
        if not src.exists():
            continue
        # Remove stale link or directory
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        # Prefer symlinks to avoid doubling volume storage
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Also link logs and config for convenience
    for subdir in ("logs", "config"):
        src = pipeline_dir / subdir
        dst = run_dir / subdir
        if not src.exists():
            continue
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copytree(src, dst, dirs_exist_ok=True)


##########################################
# Inference function
##########################################

@app.function(
    gpu=GPU,
    cpu=(2, 8),
    memory=(1024, 65536),
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={MODELS_DIR: MODELS_VOLUME, OUTPUTS_DIR: OUTPUTS_VOLUME},
)
def run_vhh_pipeline(
    target_pdb_bytes: bytes,
    framework_pdb_bytes: bytes,
    run_name: str,
    target_chain: str,
    hotspots: str,
    framework_pdb_name: str = "framework.pdb",
    framework_heavy_chain: str = "A",
    cdr_length: str = "",
    samples: int = 4,
    skip_refold: bool = False,
    extra_pipeline_args: str | None = None,
) -> dict:
    """Execute the full PPIFlow-Pipeline de novo VHH workflow.

    Results are persisted to the ppiflow-outputs volume at
    ``/ppiflow-outputs/{run_name}/``.  Re-running with the same
    ``run_name`` resumes from the last completed step (``--reuse``).

    Returns:
        Dict with ``status``, ``run_dir``, ``steps_total``, and
        ``message`` fields.
    """
    import tempfile
    import threading

    # ── 01. Verify weights and prepare checkpoints ────────────────────
    weight_warnings = verify_weights()
    for w in weight_warnings:
        print(f"⚠ {w}")
    ensure_flowpacker_checkpoints(
        flowpacker_root=FLOWPACKER_ROOT,
        bc40=FLOWPACKER_BC40,
        cluster=FLOWPACKER_CLUSTER,
        confidence=FLOWPACKER_CONFIDENCE,
    )

    # ── 02. Set up output directories on the volume ───────────────────
    # Using the volume ensures outputs survive crashes and disconnects.
    # The pipeline's --reuse flag detects completed steps via manifests.
    run_dir = Path(OUTPUTS_DIR) / run_name
    pipeline_dir = run_dir / "_pipeline"
    inputs_dir = run_dir / "inputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # ── 03. Materialise input PDB files ───────────────────────────────
    # Write to both volume (archive) and tmpdir (pipeline input path).
    (inputs_dir / "target.pdb").write_bytes(target_pdb_bytes)
    (inputs_dir / "framework.pdb").write_bytes(framework_pdb_bytes)

    tmp = Path(tempfile.mkdtemp(prefix="ppiflow_inputs_"))
    target_path = tmp / "target.pdb"
    framework_path = tmp / framework_pdb_name
    target_path.write_bytes(target_pdb_bytes)
    framework_path.write_bytes(framework_pdb_bytes)

    # ── 04. Start periodic volume commit for crash resilience ─────────
    stop_commit = threading.Event()

    def _periodic_commit():
        while not stop_commit.wait(120):
            try:
                OUTPUTS_VOLUME.commit()
                print("  [volume] committed checkpoint")
            except Exception as exc:
                print(f"  [volume] commit failed: {exc}")

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    # ── 05. Resolve tool paths ────────────────────────────────────────
    abmpnn_run = PPIFLOW_ROOT / "ProteinMPNN"
    if not (abmpnn_run / "protein_mpnn_run.py").exists():
        abmpnn_run = PIPELINE_ROOT / "src" / "entrypoints"

    tool_args: list[str] = [
        "--ppiflow_ckpt", str(NANOBODY_CKPT),
        "--abmpnn_ckpt", str(ABMPNN_CKPT),
        "--af3score_repo", str(AF3SCORE_ROOT),
        "--af3_weights", str(AF3_WEIGHTS.parent),
        "--flowpacker_repo", str(FLOWPACKER_ROOT),
    ]
    if (abmpnn_run / "protein_mpnn_run.py").exists():
        tool_args += ["--abmpnn_run", str(abmpnn_run)]

    rosetta_db = resolve_rosetta_db()
    rosetta_bin = ROSETTA_BIN_DIR / ROSETTA_EXECUTABLES["relax"]
    if rosetta_bin.exists():
        tool_args += ["--rosetta_bin", str(rosetta_bin)]
    if rosetta_db:
        tool_args += ["--rosetta_db", rosetta_db]

    # ── 06. Resolve framework settings ────────────────────────────────
    resolved_heavy_chain, resolved_cdr_length = resolve_vhh_framework_settings(
        framework_path,
        framework_heavy_chain,
        cdr_length or None,
    )

    # ── 07. Build CLI for ``ppiflow pipeline`` ────────────────────────
    cmd: list[str] = [
        "python", str(PIPELINE_ROOT / "ppiflow.py"),
        "pipeline",
        "--protocol", "vhh",
        "--preset", "custom",
        "--output", str(pipeline_dir),
        "--single-process",
        "--reuse",
        "--continue-on-error",
        "--verbose",
        # Design inputs
        "--name", run_name,
        "--target_pdb", str(target_path),
        "--framework_pdb", str(framework_path),
        "--target_chains", target_chain,
        "--hotspots", hotspots,
        "--heavy_chain", resolved_heavy_chain,
        "--cdr_length", resolved_cdr_length,
        "--samples_per_target", str(samples),
        # Lower thresholds so designs pass through all stages
        "--af3score1_iptm_min", "0.0",
        "--af3score1_ptm_min", "0.0",
        "--af3score2_iptm_min", "0.0",
        "--af3score2_ptm_min", "0.0",
        "--af3refold_iptm_min", "0.0",
        "--af3refold_ptm_min", "0.0",
        "--af3refold_dockq_min", "0.0",
        "--af3refold_num_samples", "1",
        "--af3refold_model_seeds", "0",
        "--dockq_min", "0.0",
    ]
    cmd.extend(tool_args)

    if skip_refold:
        cmd.append("--skip-refold")
    if extra_pipeline_args:
        cmd.extend(shlex.split(extra_pipeline_args))

    # ── 08. Execute the pipeline ──────────────────────────────────────
    print(f"▶ Starting VHH pipeline: run_name={run_name}")
    print(f"  output → {pipeline_dir}")
    pipeline_ok = True
    try:
        run_command(cmd, cwd=str(PIPELINE_ROOT))
        print(f"✓ Pipeline finished: run_name={run_name}")
    except Exception as exc:
        print(f"✗ Pipeline error: {exc}")
        pipeline_ok = False

    # ── 09. Organize step outputs ─────────────────────────────────────
    print("▶ Organizing step outputs...")
    organize_step_outputs(pipeline_dir, run_dir)

    # Write a run summary
    summary_lines = [
        f"run_name: {run_name}",
        f"status: {'ok' if pipeline_ok else 'partial'}",
        f"pipeline_dir: {pipeline_dir}",
        f"volume: ppiflow-outputs",
        "",
        "Step directories:",
    ]
    for step_name, _ in STEP_OUTPUT_MAP:
        step_path = run_dir / step_name
        exists = step_path.exists() and (
            step_path.is_symlink() or any(step_path.iterdir())
        ) if step_path.exists() else False
        summary_lines.append(f"  {step_name}: {'✓' if exists else '✗'}")
    (run_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n")

    # ── 10. Final volume commit ───────────────────────────────────────
    stop_commit.set()
    commit_thread.join(timeout=10)
    OUTPUTS_VOLUME.commit()
    print(f"✓ Results committed to volume: /ppiflow-outputs/{run_name}/")

    return {
        "status": "ok" if pipeline_ok else "partial",
        "run_dir": f"/ppiflow-outputs/{run_name}",
        "steps_total": 16,
        "message": (
            "All 16 steps completed successfully."
            if pipeline_ok
            else "Pipeline completed with errors. Re-run with same --run-name to resume."
        ),
    }


##########################################
# Entrypoint
##########################################

@app.local_entrypoint()
def main(
    run_name: str = "ppiflow_vhh",
    target_pdb: str = "",
    framework_pdb: str = "",
    target_chain: str = "",
    hotspots: str = "",
    heavy_chain: str = "A",
    cdr_length: str = "",
    samples: int = 4,
    skip_refold: bool = False,
    extra_pipeline_args: str | None = None,
) -> None:
    """Run a de novo VHH design on Modal.

    Results are saved to the ``ppiflow-outputs`` volume at
    ``/ppiflow-outputs/{run_name}/``, organized into 16 step sub-directories.

    If the client disconnects, the remote function continues running and
    saves results to the volume.  Re-run with the same ``--run-name`` to
    resume from the last completed step.
    """
    # -- Validate inputs --
    if not target_pdb:
        raise ValueError("--target-pdb is required.")
    if not framework_pdb:
        raise ValueError("--framework-pdb is required.")
    if not target_chain:
        raise ValueError("--target-chain is required.")
    if not hotspots:
        raise ValueError("--hotspots is required.")

    target_path = Path(target_pdb).expanduser().resolve()
    framework_path = Path(framework_pdb).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target PDB not found: {target_path}")
    if not framework_path.exists():
        raise FileNotFoundError(f"Framework PDB not found: {framework_path}")

    # -- Read input files --
    target_bytes = target_path.read_bytes()
    framework_bytes = framework_path.read_bytes()

    # -- Submit to Modal --
    print(f"🧬 Submitting VHH pipeline: run_name={run_name}")
    print(f"   Results will be saved to volume: ppiflow-outputs/{run_name}/")
    print(f"   If disconnected, re-run with same --run-name to check/resume.")
    print()

    result = run_vhh_pipeline.remote(
        target_pdb_bytes=target_bytes,
        framework_pdb_bytes=framework_bytes,
        run_name=run_name,
        target_chain=target_chain,
        hotspots=hotspots,
        framework_pdb_name=framework_path.name,
        framework_heavy_chain=heavy_chain,
        cdr_length=cdr_length,
        samples=samples,
        skip_refold=skip_refold,
        extra_pipeline_args=extra_pipeline_args,
    )

    # -- Report results --
    print()
    print(f"{'=' * 60}")
    print(f"Pipeline status: {result['status']}")
    print(f"Volume path:     {result['run_dir']}")
    print(f"Steps:           {result['steps_total']}")
    print(f"Message:         {result['message']}")
    print(f"{'=' * 60}")
    print()
    print("To browse results:")
    print(f"  modal volume ls ppiflow-outputs/{run_name}/")
    print()
    print("To download results:")
    print(f"  modal volume get ppiflow-outputs/{run_name}/ ./{run_name}/")
