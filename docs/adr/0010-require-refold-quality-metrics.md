# Require ReFold quality metrics

PPIFlow ReFold outputs include candidate-keyed quality metrics in addition to refolded structures. AlphaFold3 inference may still return its native archive, but the workflow must derive or expose a metrics table from confidence and ranking outputs so DockQ, Rosetta relax, Rank, and Report do not rely on unkeyed JSON files or structure filename ordering.
