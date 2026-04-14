"""Helper functions for structure processing."""

from pathlib import Path

import gemmi


def struct2seq(path: str | Path, **kwargs) -> list[tuple[str, str]]:
    """Convert a PDB or mmCIF file to a list of sequences.

    Each tuple contains a chain ID and its corresponding sequence.
    Note that only the first model in the structure is considered.

    WARNING: non-standard residue names and non-polymer molecules are silently dropped
        unless they are passed as additional keyword arguments. For example, to map
        "ASH" to "A", you can call this function as `struct2seq(path, ASH="A")`.
        Similarly, you can map water molecules or ligands with
        `struct2seq(path, HOH="o", LIG="l")`.

    Args:
        path: Path to the PDB or mmCIF structure file. The file format is
            automatically detected from the content, and works for both
            text and gzipped files.
        **kwargs: Additional mappings for residue names (3-letter to 1-letter).

    Returns:
        A list of tuple(chain_id, sequence).
    """
    st = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Detect)
    st.setup_entities()
    st.remove_ligands_and_waters()

    model = st[0]
    seqs: list[tuple[str, str]] = []
    for chain in model:
        chain_seq: list[str] = []
        for res in chain:
            res_name = res.name
            if res_name in kwargs:
                chain_seq.append(kwargs[res_name])
            else:
                res_info = gemmi.find_tabulated_residue(res_name)
                res_code = res_info.one_letter_code
                if res_code != " ":
                    chain_seq.append(res_code.upper())
                else:
                    chain_seq.append("X")
        seqs.append((chain.name, "".join(chain_seq)))
    return seqs
