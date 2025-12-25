#!/usr/bin/env python3
"""Create GROMACS topology and coordinate files using Amber force fields."""

# Amber to Gromacs, ParmEd
import parmed as pmd

# convert AMBER topology to GROMACS, CHARMM formats
amber = pmd.load_file("prmtop", "inpcrd")

# cmaptypes correction
# https://github.com/ParmEd/ParmEd/issues/1292
CX_dir = {
    "ALA": "XC0",
    "ARG": "XC1",
    "ASH": "XC2",
    "ASN": "XC3",
    "ASP": "XC4",
    "CYM": "XC5",
    "CYS": "XC6",
    "CYX": "XC5",
    "GLH": "XC7",
    "GLN": "XC8",
    "GLU": "XC9",
    "GLY": "XC10",
    "HID": "XC5",
    "HIE": "XC5",
    "HIP": "XC5",
    "HYP": "XC5",
    "ILE": "XC11",
    "LEU": "XC5",
    "LYN": "XC12",
    "LYS": "XC12",
    "MET": "XC5",
    "PHE": "XC5",
    "PRO": "XC13",
    "SER": "XC14",
    "THR": "XC15",
    "TRP": "XC5",
    "TYR": "XC5",
    "VAL": "XC11",
}

known_types = {}
for cmap in amber.cmaps:  # loop through all cmap terms
    atom_type = cmap.atom3.atom_type  # get atom type
    new_typename = CX_dir[cmap.atom3.residue.name]  # get new CA type name (e.g. XC0)
    if (
        new_typename in known_types
    ):  # if new type name in known types, directly assign atom to new type
        cmap.atom3.atom_type = known_types[cmap.type]
        cmap.atom3.type = known_types[cmap.type].name
        continue  # skip

    # if new type name is not in known types, create new type
    new_type = pmd.AtomType(
        new_typename,
        atom_type.number,
        atom_type.mass,
        atom_type.atomic_number,
        new_typename,
        atom_type.charge,
    )
    new_type.epsilon = atom_type.epsilon  # copy over epsilon and rmin
    new_type.rmin = atom_type.rmin
    known_types[cmap.type] = new_type

    # assgin atom to new type
    cmap.atom3.atom_type = known_types[cmap.type]
    cmap.atom3.type = known_types[cmap.type].name

# Save a GROMACS topology and GRO files
amber.save("gromacs.top")
amber.save("gromacs.gro")
