# Record workflow and app-volume candidate file locations

PPIFlow candidate manifests record both workflow-relative artifact paths and app-volume paths when both are available. Workflow-relative paths support materialized downstream artifacts and user inspection, while app-volume paths plus volume identity support strict availability checks for app-owned durable outputs without guessing storage ownership.
