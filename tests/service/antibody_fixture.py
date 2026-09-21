"""Tiny public-sequence therapeutic reference for offline service/browser tests."""

import polars as pl

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
            "HeavySequence(ifbispec)": [None, VH + "H", None],
            "LightSequence(ifbispec)": [None, VL + "H", None],
        })
        .write_csv()
        .encode()
    )
