# Fail PPIFlow candidate joins by default

PPIFlow candidate joins fail by default when required candidate identities are missing from either side. Silent drops can hide lost structures or scores in scientific workflows, so missing-candidate tolerance must be an explicit inspection/debug setting rather than the normal execution path.
