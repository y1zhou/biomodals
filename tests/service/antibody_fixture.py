"""Tiny public-sequence therapeutic reference for offline service/browser tests."""

import hashlib
import zipfile
from io import BytesIO

import orjson
import polars as pl

from biomodals.helper.antibody import assign_germlines
from biomodals.workflow.humanization.export import IMGT_MUTATION_SCHEMA
from biomodals.workflow.humanization.germlines import (
    GERMLINE_SCHEMA,
    add_gene_columns,
    add_sequence_columns,
)


def annotated_archive(
    table, run_id, settings, versions, *, tamper=False, schema_version=5
):
    """Build native-shaped annotated results without any cloud/model operation."""
    assignments = {
        sequence: assign_germlines(sequence)
        for sequence in set(table["vh"]) | set(table["vl"])
    }
    rows = [
        {
            "parent_id": row["parent_id"],
            "candidate_id": row["candidate_id"],
            "chain": chain,
            "sequence_sha256": hashlib.sha256(row[chain].encode()).hexdigest(),
            **assignments[row[chain]],
        }
        for row in table.iter_rows(named=True)
        for chain in ("vh", "vl")
    ]
    evidence = pl.DataFrame(rows, schema=GERMLINE_SCHEMA)
    table = (add_sequence_columns if schema_version == 5 else add_gene_columns)(
        table, evidence
    )
    if tamper:
        evidence = evidence.with_columns(pl.lit("incorrect").alias("sequence_sha256"))
    parquet = BytesIO()
    evidence.write_parquet(parquet)
    mutations = BytesIO()
    pl.DataFrame(schema=IMGT_MUTATION_SCHEMA).write_parquet(mutations)
    generation = BytesIO()
    pl.DataFrame(schema={"candidate_id": pl.String}).write_parquet(generation)
    members = {
        "selection.csv": table.write_csv().encode(),
        "germlines.parquet": parquet.getvalue(),
        "imgt_mutations.parquet": mutations.getvalue(),
        "generation.parquet": generation.getvalue(),
    }
    members["manifest.json"] = orjson.dumps({
        "schema_version": schema_version,
        "execution_run_id": str(run_id),
        "parameters": settings.model_dump(),
        "scientific_versions": versions,
        "candidate_count": table.height,
        "status": "succeeded",
        "files": [
            {
                "path": name,
                "size_bytes": len(content),
                "content_sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in members.items()
        ],
    })
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGGYFDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"


def reference_csv() -> bytes:
    """Exercise exact Approved (not Active) and secondary-chain inclusion."""
    return (
        pl
        .DataFrame({
            "Highest_Clin_Trial (Feb '25)": ["Approved", "Approved", "Approved (w)"],
            "Est. Status": ["Active", "NFD", "Active"],
            "HeavySequence": [VH, VH.lower(), VH + "C"],
            "LightSequence": [VL, VL.lower(), VL + "C"],
            "HeavySequence(ifbispec)": ["na", VH + "H", None],
            "LightSequence(ifbispec)": ["na", VL + "H", None],
        })
        .write_csv()
        .encode()
    )
