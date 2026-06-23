# Report PPIFlow candidate attrition

PPIFlow keeps rejected, failed, and skipped candidates available for reporting even though downstream scientific stages consume only retained manifests. This separates execution semantics from audit/reporting needs: filters narrow the active candidate set, while the final report can explain where candidates were lost.
