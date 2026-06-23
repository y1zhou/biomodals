# Split PPIFlow helper tests from workflow integration tests

Pure PPIFlow helper tests live in focused test modules separate from `tests/workflow/test_ppiflow_workflow.py`. The workflow test file keeps DAG shape, node contract, and namespace integration coverage, while manifest, table, staging, and coordinator helper tests can fail independently with clearer scope.
