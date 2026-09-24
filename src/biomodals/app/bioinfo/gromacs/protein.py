"""Protein atom selection using the pinned GROMACS residue definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import biotite.structure as struc
    import numpy as np

# Protein entries in GROMACS 2026.1 share/top/residuetypes.dat. Use the same
# classification as native Protein/C-alpha groups, not the PDB CCD: force-field
# names encode protonation, disulfides and termini and must not be renamed.
# https://github.com/gromacs/gromacs/blob/v2026.1/share/top/residuetypes.dat
PROTEIN_RESIDUES = tuple(
    """
ABU ACE AIB ALA ARG ARGN ASN ASN1 ASP ASP1 ASPH ASPP ASH CT3 CYS CYS1 CYS2
CYSH DALA GLN GLU GLUH GLUP GLH GLY HIS HIS1 HISA HISB HISH HISD HISE HISP
HSD HSE HSP HYP ILE LEU LSN LYS LYSN LYSH MELEU MET MEVAL NAC NME NHE NH2
PHE PHEH PHEU PHL PRO SER THR TRP TRPH TRPU TYR TYRH TYRU VAL PGLU HID HIE
HIP LYP LYN CYN CYM CYX DAB ORN NALA NGLY NSER NTHR NLEU NILE NVAL NASN
NGLN NARG NHID NHIE NHIP NHISD NHISE NHISH NTRP NPHE NTYR NGLU NASP NLYS
NORN NDAB NLYSN NPRO NHYP NCYS NCYS2 NMET NASPH NGLUH CALA CGLY CSER CTHR
CLEU CILE CVAL CASN CGLN CARG CHID CHIE CHIP CHISD CHISE CHISH CTRP CPHE
CTYR CGLU CASP CLYS CORN CDAB CLYSN CPRO CHYP CCYS CCYS2 CMET CASPH CGLUH
""".split()
)


class NonProteinTemplateError(ValueError):
    """The retained template includes residues outside native Protein selection."""


def protein_mask(atoms: struc.AtomArray) -> np.ndarray:
    """Select native protein atoms without changing names, order or coordinates."""
    import numpy as np

    return np.isin(atoms.res_name, PROTEIN_RESIDUES)


def require_protein(atoms: struc.AtomArray) -> None:
    """Reject non-protein clustering templates with a safe, actionable error."""
    if not protein_mask(atoms).all():
        raise NonProteinTemplateError(
            "The retained trajectory template contains residues outside the "
            "GROMACS Protein group. Clustering requires the matching "
            "protein-only processed trajectory and template."
        )
