# Antibody analysis acknowledgements

The potential sequence-liability detector in `biomodals.helper.antibody`
adapts the active rules of [LAMBS v0.12.0](https://github.com/dcroote/lambs/tree/61d6f28f3c666778fc06ed05a8a2d50faef4d715)
by Daniel Croote and contributors. The Python implementation uses explicit,
overlap-preserving input intervals and Biopython's Kyte–Doolittle constants;
it does not embed the LAMBS web application. The upstream
[Apache-2.0 license](LAMBS-LICENSE.txt) is retained alongside this notice.

[Arpeggia](https://github.com/y1zhou/arpeggia/tree/v0.10.1) is distributed under
GPL-3.0. Its bundled germline references include IMGT data with their own
[attribution and terms](https://github.com/y1zhou/arpeggia/blob/v0.10.1/data/germlines/README.md).
These dependency and reference notices remain applicable; this project does
not relicense the upstream data.

Therapeutic reference frequencies derive from
[Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/about/).
The service downloads the public CSV to build an operator-owned derived cache;
that CSV is not bundled in this repository. Its download time, digest and cohort are shown
with the analysis. See the database's citation guidance when using these data.
