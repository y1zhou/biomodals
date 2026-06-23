# Keep the PPIFlow manifest schema workflow-local

The PPIFlow candidate manifest schema starts as workflow-local helpers rather than shared `biomodals.schema` models. It is currently a PPIFlow-specific provenance contract, and moving it into shared schemas should wait until another workflow needs the same candidate-manifest abstraction.
