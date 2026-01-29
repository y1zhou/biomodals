"""
RFdiffusion on Modal - minimal runnable scaffold (fixed).

Typical usage:

  # 1) download models into the Modal Volume (run once)
  modal run rfdiffusion_app.py --download-models --force-redownload

  # 2) run inference (binder design / scaffold etc.)
  modal run rfdiffusion_app.py \
    --run-name test1 \
    --input-pdb ~/outputs/RFdiffusion/RBD_wt.pdb \
    --rfd-args "contigmap.contigs='[100-150/0 E333-526]' inference.num_designs=2 ppi.hotspot_res=[E405,E408]"
"""

import os
from pathlib import Path

from modal import App, Image, Volume


# -------------------------
# Modal configs
# -------------------------
GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", "36000"))
APP_NAME = os.environ.get("MODAL_APP", "RFdiffusion")
RFD_VOLUME = Volume.from_name("rfdiffusion-models", create_if_missing=True)
RFD_OUT_VOLUME = Volume.from_name("rfdiffusion-outputs", create_if_missing=True)

RFD_MODELS_DIR = "/root/rfdiffusion_models"
RFD_REPO_DIR = "/root/RFdiffusion"


# -------------------------
# Image definition
# -------------------------
runtime_image = (
    Image.micromamba(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "ca-certificates",
        "zstd",
        "tar",
        "findutils",  # for `find`
    )
    .env(
        {
            "UV_TORCH_BACKEND": "cu128",
            "PYTHONUNBUFFERED": "1",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",  # 关键：切回 weights_only=False（当调用方未显式传参时）
        }
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git {RFD_REPO_DIR}"
    )
    .uv_pip_install(
        "torch",
        "torchdata>=0.7",     # ✅ NEW：给 dgl 用的 datapipes
        "numpy",
        "scipy",
        "tqdm",
        "pyyaml",
        "omegaconf",
        "hydra-core",
        "biopython",
        "pandas",
        "einops",
        "opt_einsum",   # ✅ CHANGED: 修复当前报错
        "dm-tree",      # （可选但推荐）
        "pyrsistent",   # ✅ NEW: RFdiffusion symmetry 依赖     
    )
    .run_commands(
    # 进入 RFdiffusion 自带的 NVIDIA SE3Transformer 目录安装
    f"bash -lc 'cd {RFD_REPO_DIR}/env/SE3Transformer && python -m pip install --no-cache-dir -r requirements.txt'",
    f"bash -lc 'cd {RFD_REPO_DIR}/env/SE3Transformer && python setup.py install'",

     # ✅ 构建期自检：环境变量 + e3nn import（关键）
    "bash -lc 'python -c \"import os,torch; print(os.environ.get(\\\"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD\\\")); import e3nn.o3; print(\\\"e3nn import OK\\\")\"'",

    # 构建期自检：必须能 import
    "bash -lc 'python -c \"import se3_transformer, sys; print(se3_transformer.__file__); print(sys.executable)\"'",  

    # 先卸载可能已经装上的 CPU dgl
    "bash -lc 'python -m pip uninstall -y dgl || true'",

    # 从 DGL 官方 wheel 仓库安装 CUDA 12.1 版本（最常用、最稳定）
    # 注意：这里安装的包名仍然是 dgl，但会拿到带 +cu121 的 wheel
    "bash -lc 'python -m pip install --no-cache-dir \"dgl==1.1.3\" -f https://data.dgl.ai/wheels/cu121/repo.html'",

)
    # ### CHANGED: 在镜像层就设置 MODELS_PATH（等下会被 Volume 挂载覆盖目录内容）
    .env({"MODELS_PATH": RFD_MODELS_DIR})
)

app = App(APP_NAME, image=runtime_image)


# -------------------------
# Helpers
# -------------------------
def run_command(
    cmd: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and stream stdout/stderr."""
    import subprocess as sp

    print("Running:", " ".join(cmd))
    with sp.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        bufsize=1,
        encoding="utf-8",
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
        rc = p.wait()
        if rc != 0:
            raise sp.CalledProcessError(rc, cmd)


def package_dir_to_tar_zst(dir_path: str) -> bytes:
    """Package a directory into tar.zst and return bytes."""
    import subprocess as sp
    from pathlib import Path

    dp = Path(dir_path)
    parent = str(dp.parent)
    name = dp.name
    cmd = ["tar", "--zstd", "-cf", "-", name]
    return sp.check_output(cmd, cwd=parent)


def build_runtime_env() -> dict[str, str]:
    """
    ### CHANGED: 统一构造运行时环境变量
    关键：把 RFdiffusion repo 加入 PYTHONPATH，解决 `import rfdiffusion` 找不到的问题
    """
    env = dict(os.environ)
    # 让 Python 能 import /root/RFdiffusion/rfdiffusion
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{RFD_REPO_DIR}:{existing}" if existing else RFD_REPO_DIR

    # 明确告诉程序权重在哪（Volume 会挂载到这个目录）
    env["MODELS_PATH"] = RFD_MODELS_DIR

    # ✅ Force torch.load default behavior (PyTorch 2.6+)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    # (Optional) quiet DGL backend warning
    env["DGLBACKEND"] = "pytorch"

    return env


# -------------------------
# Step 1: download model weights into the Volume
# -------------------------
@app.function(
    cpu=(2, 8),
    timeout=TIMEOUT * 2,
    volumes={RFD_MODELS_DIR: RFD_VOLUME},
)
def download_rfdiffusion_models(force: bool = False) -> None:
    """
    Download RFdiffusion model weights into the mounted Volume.

    你必须根据你们团队使用的 RFdiffusion 版本，落实“权重下载”方式：
    - 有的仓库带 scripts/download_models.sh
    - 有的需要手工下载 ckpt
    """
    env = build_runtime_env()

    # Ensure target dir exists
    run_command(["bash", "-lc", f"mkdir -p {RFD_MODELS_DIR}"], env=env)

    download_script = f"{RFD_REPO_DIR}/scripts/download_models.sh"

    # ### CHANGED: 不再拼长 bash 字符串做复杂逻辑，尽量简洁可控
    if force:
        print("force=True: you may want to delete old weights first (optional).")

    # 这个逻辑保留：如果仓库有脚本就跑，没有就直接失败并提示你补齐下载方式
    bash = [
        "bash",
        "-lc",
        f"""
set -e
if [ -f "{download_script}" ]; then
  echo "Found download script: {download_script}"
  bash "{download_script}" "{RFD_MODELS_DIR}"
else
  echo "No download_models.sh found in this RFdiffusion repo."
  echo "You must implement weights download for your RFdiffusion version."
  echo "Expected weights under: {RFD_MODELS_DIR}"
  exit 1
fi
""",
    ]
    run_command(bash, env=env)

    RFD_VOLUME.commit()
    print("RFdiffusion model download complete and committed.")


# -------------------------
# Step 2: inference function (remote GPU job)
# -------------------------
@app.function(
    gpu=GPU,
    cpu=(2, 16),
    memory=(4096, 65536),
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={RFD_MODELS_DIR: RFD_VOLUME.read_only()},
)
def rfdiffusion_infer(
    input_pdb_bytes: bytes,
    input_pdb_name: str,
    run_name: str,
    rfd_args: str,
) -> bytes:
    """
    Run RFdiffusion inference inside the container.
    """
    import shlex
    from tempfile import TemporaryDirectory

    env = build_runtime_env()  # ### CHANGED

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        input_pdb = tmp / input_pdb_name
        input_pdb.write_bytes(input_pdb_bytes)

        out_dir = tmp / f"{run_name}_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # ✅ Runtime GPU / DGL CUDA sanity check (runs inside GPU job)
        # =====================================================
        run_command(
            [
                "bash",
                "-lc",
                "python -c \""
                "import torch, dgl; "
                "print('torch', torch.__version__); "
                "print('torch cuda', torch.cuda.is_available()); "
                "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
                "print('dgl', dgl.__version__); "
                "import sys; "
                "sys.exit(0) if torch.cuda.is_available() else sys.exit(2)\"",
            ],
            env=env,
        )

        # Only test DGL->CUDA if CUDA is available
        run_command(
            [
                "bash",
                "-lc",
                "python -c \""
                "import torch, dgl; "
                "assert torch.cuda.is_available(); "
                "g=dgl.graph(([0],[1])); "
                "g=g.to('cuda'); "
                "print('dgl graph device', g.device)\"",
            ],
            env=env,
        )

        run_infer_py = f"{RFD_REPO_DIR}/scripts/run_inference.py"

        # ### CHANGED (关键): 用 shlex.split 正确切分参数，避免空格/[] 引号问题
        # 例如：contigmap.contigs=[10-50/0 E333-526] 这种带空格的 token，
        # 以前在 bash 里很容易被拆开导致报错；现在不会。
        extra_tokens = shlex.split(rfd_args) if rfd_args else []

        # =====================================================
        # ✅ Check model weights in Volume and copy to repo default path
        # =====================================================
        default_models_dir = f"{RFD_REPO_DIR}/models"
        ckpt_src = f"{RFD_MODELS_DIR}/Complex_base_ckpt.pt"      # Volume mount
        ckpt_dst = f"{default_models_dir}/Complex_base_ckpt.pt"  # RFdiffusion default

        # 1) list what we have in the mounted Volume
        run_command(
            ["bash", "-lc", f"echo '=== MODELS VOLUME ({RFD_MODELS_DIR}) ==='; ls -lah {RFD_MODELS_DIR}"],
            env=env,
        )

        # 2) fail fast if checkpoint missing
        run_command(
            ["bash", "-lc", f"test -f {ckpt_src} && ls -lh {ckpt_src} || (echo 'MISSING ckpt: {ckpt_src}'; exit 1)"],
            env=env,
        )

        # 3) copy into repo default models dir (what RFdiffusion actually uses)
        run_command(["bash", "-lc", f"mkdir -p {default_models_dir}"], env=env)
        run_command(
            ["bash", "-lc", f"cp -f {ckpt_src} {ckpt_dst} && echo '=== COPIED CKPT ===' && ls -lh {ckpt_dst}"],
            env=env,
        )

        # ===============================
        # 正式运行 RFdiffusion
        # ===============================

        # ### CHANGED: 直接用 subprocess 列表参数调用 python（不再用 bash -lc 拼字符串）
        cmd = [
            "python",
            run_infer_py,
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={out_dir}/rfout",
            *extra_tokens,
        ]

        # 在 repo 目录下运行（让相对路径/配置更稳定）
        run_command(cmd, cwd=RFD_REPO_DIR, env=env)

        # optional: list outputs
        run_command(
            ["bash", "-lc", f'echo "=== outputs ==="; find "{out_dir}" -maxdepth 3 -type f | head -n 200'],
            env=env,
        )

        tar_bytes = package_dir_to_tar_zst(str(out_dir))
        return tar_bytes


# -------------------------
# Local entrypoint (CLI)
# -------------------------
@app.local_entrypoint()
def main(
    run_name: str | None = None,
    input_pdb: str | None = None,
    rfd_args: str = "",
    download_models: bool = False,
    force_redownload: bool = False,
    out_dir: str | None = None,
):
    if download_models:
        download_rfdiffusion_models.remote(force=force_redownload)
        return

    if run_name is None:
        raise ValueError("Missing required --run-name")
    if input_pdb is None:
        raise ValueError("Missing required --input-pdb (path to local .pdb)")

    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    pdb_bytes = input_path.read_bytes()

    local_out = Path(out_dir).expanduser().resolve() if out_dir else Path.cwd()
    local_out.mkdir(parents=True, exist_ok=True)
    out_file = local_out / f"{run_name}_rfdiffusion_outputs.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output already exists: {out_file}")

    tar_bytes = rfdiffusion_infer.remote(
        input_pdb_bytes=pdb_bytes,
        input_pdb_name=input_path.name,
        run_name=run_name,
        rfd_args=rfd_args,
    )
    out_file.write_bytes(tar_bytes)
    print(f"Done. Saved: {out_file}")
