# p-AbNatiV2 upstream compatibility patches

The p-AbNatiV2 app targets AbNatiV 2.0.8 at commit
`eb517f1f0b947084cb7e44a54ef34103e9692f5e` and ABodyBuilder3 at commit
`18e4058015a39c5405c08a0d5629cf302627b253`. Its Modal image applies two
guarded compatibility edits after installing those exact sources:

- change one of two adjacent `0.25` coordinates in AbNatiV's
  `camsol_strucorr_colormap` to `0.250001`, because current Matplotlib rejects
  equal adjacent coordinates while importing the module; and
- index with `data[tuple(ranges)]` rather than `data[ranges]` in
  ABodyBuilder3's OpenFold tensor utility, as required by current NumPy. This is
  the compatibility edit documented by p-AbNatiV2 upstream for Python 3.12.

Both patches require the exact original source text, are idempotent, and fail
the image build if upstream changes invalidate their preconditions. Neither
changes model parameters, inputs, scoring formulas, mutation selection, or
structure-model inference. The app does not patch scoring or humanization
logic.

The AbNatiV 2.0.8 distribution wheel also omits the six nested humanization
PSSM files because its package-data declaration addresses the wrong package
directory. The image restores only those files from the exact pinned commit
and verifies a fixed SHA-256 digest for each one. This is missing package data,
not a source edit; the checksums are recorded in every result manifest.

These compatibility patches are part of the app's pinned runtime identity.
Any additional upstream patch requires a separate equivalence review.
