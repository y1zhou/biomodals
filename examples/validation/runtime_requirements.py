"""Export the same helper dependencies used by mounted app images."""

import argparse

from biomodals.helper import helper_dependencies

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--relax-versions", action="store_true")
args = parser.parse_args()
# These runtime checks do not import uniaf3, which requires Python 3.12.
print(
    "\n".join(
        helper_dependencies(
            skip_deps=["uniaf3"], ignore_dep_versions=args.relax_versions
        )
    )
)
