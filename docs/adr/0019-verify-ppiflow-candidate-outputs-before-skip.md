# Verify PPIFlow candidate outputs before retry skip

PPIFlow remote stage coordinators verify a completed candidate's expected output files before skipping it on retry. Candidate manifest rows are durable provenance, but they do not replace artifact availability checks; if expected files are missing, the candidate is treated as incomplete and rerun or failed according to the stage status rules.
