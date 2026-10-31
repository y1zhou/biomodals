# Caliby position-wise constraints

Caliby supports applying residue-level constraints during sequence design via
the `pos_constraint_csv` argument. This CSV is used by both single-structure
sequence design and ensemble-conditioned sequence design. It is not an
ensemble-generation setting.

The format below follows the Caliby README section "Position-wise constraints".
For upstream examples, see:

- `github/caliby/examples/example_data/pos_constraint_csvs/native_pdb_constraints.csv`
- `github/caliby/examples/scripts/seq_des_constraints.sh`
- `github/caliby/examples/scripts/seq_des_ensemble_constraints.sh`

## CSV columns

The CSV must contain a `pdb_key` column and may contain a subset of the optional
constraint columns below.

| Column name | Format example | Description |
|-------------|----------------|-------------|
| `pdb_key` | `7xhz` | The PDB/CIF file stem, meaning the filename without extension. For `7xhz.cif`, use `7xhz`. For ensemble-conditioned sequence design, this should be the ensemble subdirectory name. |
| `fixed_pos_seq` | `A1-100,B1-100` | Sequence positions in the input structure to condition on so that they remain fixed during design. For ensemble-conditioned design, these fixed sequence identities are taken from the primary conformer. |
| `fixed_pos_scn` | `A1-10,A12,A15-20` | Sidechain positions in the input structure to condition on so that their sidechain coordinates are fixed during design. These positions should be a subset of `fixed_pos_seq`, because fixing a sidechain without also fixing its residue identity is not meaningful. |
| `fixed_pos_override_seq` | `A26:A,A27:L` | Positions where Caliby first overwrites the input residue identity with a specified amino acid, and then fixes that position during design. The colon separates the position and the desired amino acid. |
| `pos_restrict_aatype` | `A26:AVG,A27:VG` | Positions where Caliby may redesign the residue, but only from a specified allowed amino acid set. The colon separates the position and the allowed amino acids. |
| `symmetry_pos` | `A10,B10,C10\|A11,B11,C11` | Symmetry groups for tying sampling across residue positions. The pipe separates groups. In this example, A10/B10/C10 are sampled together, and A11/B11/C11 are sampled together. This column is used by Caliby's homooligomer examples and is not used in `native_pdb_constraints.csv`. |

If `pos_constraint_csv` is not provided, Caliby redesigns all positions. If an
optional column is absent, Caliby does not apply that constraint type. If a
column is present but the value is empty for a given `pdb_key`, Caliby skips
that constraint type for that target.

Residue positions should generally use the mmCIF `label_seq_id` numbering, not
`auth_seq_id`. In PyMOL, run `set cif_use_auth, off` before loading the CIF if
you want to view positions using `label_seq_id`.

## What is the primary conformer?

For ensemble-conditioned design, Caliby expects a top-level conformer directory
with one subdirectory per target. Each subdirectory name is the `pdb_key`.
Inside each subdirectory:

- The primary conformer is the original representative structure named
  `<PDB_KEY>.pdb` or `<PDB_KEY>.cif`.
- Additional `.pdb` or `.cif` files in the same subdirectory are treated as
  extra conformers and are ordered by natural alphabetical order.

The primary conformer is included by default when Caliby builds the ensemble.
It is also the sequence reference for constraints such as `fixed_pos_seq`, and
for ensemble scoring Caliby scores the sequence corresponding to the primary
conformer while ignoring sequences from additional conformers.

## What does `fixed_pos_override_seq` mean?

`fixed_pos_override_seq` means "replace first, then fix".

For example:

```text
A36:C,A37:C,A38:C,A39:C,A40:C
```

means:

1. Treat chain A residues 36-40 as cysteine (`C`) in the input sequence.
2. Add those positions to the fixed-sequence mask.
3. During design, Caliby conditions on those cysteines rather than redesigning
   those positions.

This is different from `pos_restrict_aatype`. With `pos_restrict_aatype`, the
position can still be sampled, but only from the allowed amino acids. With
`fixed_pos_override_seq`, the specified amino acid is imposed and fixed.

## Example file

`native_pdb_constraints.csv` contains:

```csv
pdb_key,fixed_pos_seq,fixed_pos_scn,fixed_pos_override_seq,pos_restrict_aatype
7xhz,"A6-15,A20-50","A6-15",,
7xz3,,,"A36:C,A37:C,A38:C,A39:C,A40:C",
8huz,,,,"A6:QR,A7:QR,A8:QR,A9:QR,A10:QR,A11:QR"
```

| Row | Meaning |
|-----|---------|
| `7xhz` | Keep sequence identity fixed at chain A residues 6-15 and 20-50. Also condition on sidechain coordinates at chain A residues 6-15. |
| `7xz3` | Override chain A residues 36-40 to cysteine (`C`) and fix those positions during design. |
| `8huz` | At chain A residues 6-11, allow only glutamine (`Q`) or arginine (`R`) during sampling. |
