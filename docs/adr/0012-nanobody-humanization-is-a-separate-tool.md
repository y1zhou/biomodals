# Keep nanobody humanization separate from paired humanization

Nanobody humanization has a standalone Tool and CLI workflow because its
single-domain candidates and VHH-compatibility objectives do not satisfy the
paired workflow's VH-VL and pairing-score contracts. Reuse the Job lifecycle,
execution kernel and applicable analysis/UI components, but do not invent a
light chain or reinterpret existing paired publications; the accepted design
is recorded in the [nanobody specification](../specs/nanobody-humanization.md).
