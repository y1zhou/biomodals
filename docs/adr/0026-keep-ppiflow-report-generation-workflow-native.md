# Keep PPIFlow report generation workflow-native

PPIFlow report generation stays a workflow-native transform that renders Markdown and HTML from materialized tables, manifests, and score artifacts. It has no expensive external runtime today, is easy to unit test, and should move to a separate app only if report rendering later needs heavyweight dependencies.
