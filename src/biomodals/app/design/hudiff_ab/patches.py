"""Guarded inference-only patches for pinned HuDiff v1.0.0."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from shutil import copyfile

SOURCE_ROOT = Path("/opt/HuDiff")


def patch_identity() -> str:
    """Fingerprint the exact guarded patch and copied inference runtime."""
    from biomodals.app.design.hudiff_ab import upstream_runtime

    digest = sha256()
    for path in (Path(__file__), Path(upstream_runtime.__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if new in source:
        if old in source:
            raise RuntimeError(f"Found mixed HuDiff patch state in {path}")
        return
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one pinned HuDiff preimage in {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def apply_hudiff_inference_patches() -> None:
    """Install the narrow runtime and remove unused inference import edges."""
    from biomodals.app.design.hudiff_ab import upstream_runtime

    sampler = SOURCE_ROOT / "antibody_scripts/sample_for_anti_cdr.py"
    sampler_prefix = """import os.path

import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from anarci import anarci, number
from copy import deepcopy
import re
from Bio import SeqIO
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)
"""
    patched_prefix = """import os.path
import sys

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

if __name__ == '__main__' and '--biomodals-input-json' in sys.argv:
    from biomodals_hudiff_runtime import main as _biomodals_main

    _biomodals_main()
    raise SystemExit(0)

import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from anarci import anarci, number
from copy import deepcopy
import re
from Bio import SeqIO
"""
    _replace_once(sampler, sampler_prefix, patched_prefix)
    copyfile(
        Path(upstream_runtime.__file__),
        SOURCE_ROOT / "antibody_scripts/biomodals_hudiff_runtime.py",
    )

    model = SOURCE_ROOT / "model/encoder/model.py"
    _replace_once(
        model,
        """from pymol import cmd
from abnumber import Chain

# Abnativ
from ..nanoencoder.abnativ_scoring import get_abnativ_nativeness_scores
""",
        "# Biomodals: unused PyMOL, AbNumber, and AbNatiV imports omitted.\n",
    )
    for old, new in (
        (
            "F.dropout(h_e, self.dropout)",
            "F.dropout(h_e, self.dropout, training=self.training or "
            "os.environ.get('HUDIFF_UPSTREAM_INFERENCE_DROPOUT', '1') == '1')",
        ),
        (
            "F.dropout(l_e, self.dropout)",
            "F.dropout(l_e, self.dropout, training=self.training or "
            "os.environ.get('HUDIFF_UPSTREAM_INFERENCE_DROPOUT', '1') == '1')",
        ),
        (
            "F.dropout(h_s)",
            "F.dropout(h_s, training=self.training or "
            "os.environ.get('HUDIFF_UPSTREAM_INFERENCE_DROPOUT', '1') == '1')",
        ),
        (
            "F.dropout(l_s)",
            "F.dropout(l_s, training=self.training or "
            "os.environ.get('HUDIFF_UPSTREAM_INFERENCE_DROPOUT', '1') == '1')",
        ),
    ):
        _replace_once(model, old, new)

    training = SOURCE_ROOT / "antibody_scripts/sample.py"
    _replace_once(
        training,
        "from patent_eval import cal_all_preservation\n",
        "# Biomodals: unused patent evaluation import omitted.\n",
    )
    _replace_once(
        training,
        "from utils.train_utils import model_selected\n",
        "# Biomodals: unused training model selector omitted.\n",
    )
    _replace_once(
        training,
        "from utils.misc import get_new_log_dir, get_logger, seed_all\n",
        "# Biomodals: unused training logging helpers omitted.\n",
    )
    _replace_once(
        training,
        "from dataset.oas_pair_dataset_new import HEAVY_REGION_INDEX, LIGHT_REGION_INDEX\n",
        "from biomodals_hudiff_runtime import HEAVY_REGION_INDEX, LIGHT_REGION_INDEX\n",
    )
    preprocess = SOURCE_ROOT / "dataset/preprocess.py"
    _replace_once(
        preprocess,
        """import pickle
import os
import json
import logging
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import pandas as pd
from tqdm import tqdm
import tempfile

from dataset.abnativ_alignment.align_and_clean import anarci_alignments_of_Fv_sequences
from dataset.abnativ_alignment.mybio import get_SeqRecords
from utils.anti_numbering import get_seq_list_from_SeqRecords
""",
        """import json
import logging
import os
import sys
import tempfile

import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)
""",
    )
