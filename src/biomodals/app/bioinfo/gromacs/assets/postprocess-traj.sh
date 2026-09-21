#!/usr/bin/env bash
# Gromacs trajectory analysis
# Adapted By Yi Zhou from script by ziwei pang (Jeffery) @biomap
# https://manual.gromacs.org/current/user-guide/terminology.html#suggested-workflow
set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi
# CLI argument parsing
DOCSTRING="usage: ./$(basename "${0}")

required arguments:
  -t, --tpr-file         Gromacs .tpr file as input
  -x, --xtc-file         Gromacs .xtc trajectory file
  -o, --output-file      Filename for the output .xtc trajectory

optional flags:
  -s, --ref-structure    Reference structure used for fitting the trajectory
  -r, --rmsd             Calculate protein backbone RMSD
  --rmsf                 Calculate the RMSF of the protein
  -d, --distance         Calculate minimum d(Ab, Ag)
"

calculate_rmsd=0
calculate_rmsf=0
calculate_dist=0
if [ -x "$(command -v 'gmx')" ]; then
  GROMACS='gmx'
elif [ -x "$(command -v 'gmx_mpi')" ]; then
  GROMACS='gmx_mpi'
else
  echo 'Neither gmx nor gmx_mpi is in PATH' >&2
  exit 1
fi
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "${DOCSTRING}"
      exit 1
      ;;
    -t|--tpr-file)
      tpr_file=$(realpath "${2}")
      ref_structure="${tpr_file}"
      shift # past argument
      shift # past value
      ;;
    -x|--xtc-file)
      xtc_file=$(realpath "${2}")
      shift
      shift
      ;;
    -o|--output-file)
      output_file=$(realpath "${2}")
      shift
      shift
      ;;
    -s|--ref-structure)
      # shellcheck disable=SC2034
      ref_structure="${2}"
      shift
      shift
      ;;
    -r|--rmsd)
      calculate_rmsd=1
      shift
      ;;
    --rmsf)
      calculate_rmsf=1
      shift
      ;;
    -d|--distance)
      calculate_dist=1
      shift
      ;;
    *)
      echo "Use ./$(basename "${0}") --help to see the arguments"
      exit 1
  esac
done

# Center the protein in the simulation box
out_dir="$(dirname "${output_file}")"
output_stem="$(basename "${output_file}" .xtc)"
mkdir -p "${out_dir}"
# printf "Protein\nProtein\nSystem\n" | \
# "${GROMACS}" trjconv \
#     -s "${tpr_file}" \
#     -f "${xtc_file}" \
#     -o "${output_file}.tmp.xtc" \
#     -pbc cluster \
#     -ur compact \
#     -center

# printf "Protein\nProtein\n" | \
# "${GROMACS}" trjconv \
#     -s "${tpr_file}" \
#     -f "${output_file}.tmp.xtc" \
#     -o "${output_file}" \
#     -tu ns \
#     -pbc mol \
#     -ur compact \
#     -boxcenter tric \
#     -center

# rm "${output_file}.tmp.xtc" && mv "${output_file}" "${output_file}.tmp.xtc"

# Re-fit to the first frame
# printf "Backbone\nProtein\n" | \
#   "${GROMACS}" trjconv \
#     -s "${ref_structure}" \
#     -f "${output_file}.tmp.xtc" \
#     -o "${output_file}" \
#     -tu ns \
#     -fit 'rot+trans'

# rm "${output_file}.tmp.xtc"

##########################################
# Extract protein trajectory, center, and remove PBC
##########################################
if [ ! -f "${output_file}" ]; then

# 1. make molecules whole
printf "Protein\nProtein\n" | \
    "${GROMACS}" trjconv -tu ns -s "${tpr_file}" \
    -f "${xtc_file}" -o "${output_file}.whole.xtc" -pbc whole

# 2. cluster molecules
printf "Protein\nProtein\n" | \
    "${GROMACS}" trjconv -tu ns -s "${tpr_file}" \
    -f "${output_file}.whole.xtc" -o "${output_file}.clust.xtc" \
    -pbc cluster -ur compact && \
    rm "${output_file}.whole.xtc"

# 3. remove jumps by using the first frame as reference
# The structure in the tpr file could have PBC so we don't use that
# "${GROMACS}" editconf -f "${tpr_file}" -o "${out_dir}/tpr_1st_frame.pdb"
printf "Protein\n" | \
"${GROMACS}" trjconv -tu ns -s "${tpr_file}" \
    -f "${output_file}.clust.xtc" -o "${output_file}.1st.pdb" \
    -dump 0 && \
printf "Protein\nProtein\n" | \
    "${GROMACS}" trjconv -tu ns -s "${output_file}.1st.pdb" \
    -f "${output_file}.clust.xtc" -o "${output_file}.nojump.xtc" \
    -pbc nojump -ur compact && \
printf "Backbone\nProtein\n" | \
    "${GROMACS}" trjconv -tu ns -s "${output_file}.1st.pdb" \
    -f "${output_file}.nojump.xtc" -o "${output_file}.fit1st.xtc" \
    -fit 'rot+trans' -ur compact && \
    rm "${output_file}.1st.pdb" "${output_file}.clust.xtc" "${output_file}.nojump.xtc"

# 4. center the system; no `-pbc nojump` after this step!
printf "Protein\nProtein\n" | \
    "${GROMACS}" trjconv -tu ns -s "${tpr_file}" \
    -f "${output_file}.fit1st.xtc" -o "${output_file}" \
    -center -ur compact -pbc mol -boxcenter tric && \
    rm "${output_file}.fit1st.xtc"
fi

if [ ! -f "${output_stem}_centered.pdb" ]; then

echo 'Protein' | "${GROMACS}" trjconv -s "${tpr_file}" -f "${output_file}" -dump 0 -o "${output_stem}_centered.gro"
echo 'Protein' | "${GROMACS}" trjconv -s "${tpr_file}" -f "${output_file}" -dump 0 -o "${output_stem}_centered.pdb"

fi

##########################################
# Downstream analyses
##########################################
# Calculate the protein backbone RMSD
if [ "${calculate_rmsd}" -eq 1 ]; then
  if [ ! -f "${output_stem}_rmsd.xvg" ]; then
  printf 'Backbone\nBackbone\n' | \
    "${GROMACS}" rms -tu ns \
        -f "${output_file}" \
        -s "${output_stem}_centered.pdb" \
        -o "${output_stem}_rmsd.xvg"
  fi
fi

# Calculate the protein RMSF
if [ "${calculate_rmsf}" -eq 1 ]; then
  if [ ! -f "${output_stem}_rmsf.xvg" ]; then
  printf 'Protein\n' | \
    "${GROMACS}" rmsf -res \
        -s "${output_stem}_centered.pdb" \
        -f "${output_file}" \
        -o "${output_stem}_rmsf.xvg" \
        -oq "${output_stem}_bfac.pdb" \
        -od "${output_stem}_rmsd.xvg"
  fi
fi

# Calculate minimum distances between Ab and Ag
if [ "${calculate_dist}" -eq 1 ]; then
  printf 'chain A or chain B\nchainC\nq\n' | \
    "${GROMACS}" make_ndx \
        -f "${output_stem}_initial_structure.pdb" \
        -o "${output_stem}_chains.ndx"

  printf 'chAORCHAINB\nchC\n' | \
    "${GROMACS}" mindist -tu ns \
        -f "${output_file}" \
        -s "${output_stem}_centered.gro" \
        -n "${output_stem}_chains.ndx" \
        -od "${output_stem}_mindist.xvg"
fi
