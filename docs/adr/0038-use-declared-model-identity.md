# Use declared model identity

Status: accepted.

Inference Run Identity includes a code-owned checkpoint label together with the
pinned AlphaFold repository and app identity. The app does not scan
`af3.bin`, compute its digest, or run a separate model-identity function before
cache reconciliation.

This decision treats the checkpoint beneath the declared model path as
immutable in place. Replacing its contents without changing the declared label
can reuse Seed Predictions created with the previous weights. An intentional
checkpoint replacement must therefore bump the code-owned model identity or
explicitly clear the affected inference run cache.

The simpler contract avoids an additional model-Volume read on every
invocation and accepts operator-controlled checkpoint immutability as the trust
boundary.
