#!/usr/bin/env bash
# author: Ziwei Pang (Jeffery), Chuang Yu, Yi Zhou @biomap

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
  set -o xtrace
fi

# CLI argument parsing
DOCSTRING="usage: ./$(basename "${0}")

required arguments:
  -i, --input-pdb        input PDB file for preparation

options:
  -h, --help             show this help message and exit

  -t, --simulation-time  the total simulation time in nanoseconds (default 100)
  -f, --fix-pdb          fix input PDB file with pdbfixer (default false)

  --cpu-only             flag to use disable GPU usage (default use GPU)
  -g, --gpu-id           integer GPU ID used to run the simulation (default 0)
  --pin                  flag to turn on affinity of threads to cores
  -j, --threads          number of CPU threads to use (default 27)
  --pin-stride           core stride to pin threads
  --pin-offset           offset cores to pin threads
  --use-gmx-mpi          use gmx_mpi instead of gmx (default false)
  --use-openmp-threads   specify threads with -ntomp instead of -nt (default false)

  --ld-seed              use a fixed ld-seed (default -1)
  --gen-seed             seed to initialize rng for random velocities (default -1)
  --genion-seed          seed for gmx genion random number generator (default 0)
"

simulation_ns=100
fix_pdb=0
cpu_only=0
gpu_id=0
num_threads=27
pin='auto'
pin_stride=0
pin_offset=0
use_gmx_mpi=0
use_openmp=0

ld_seed='-1'
gen_seed='-1'
genion_seed=0
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    echo "${DOCSTRING}"
    exit 1
    ;;
  -i | --input-pdb)
    pdb_file="${2}"
    shift # past argument
    shift # past value
    ;;
  -t | --simulation-time)
    simulation_ns="${2}"
    shift
    shift
    ;;
  -f | --fix-pdb)
    fix_pdb=1
    shift
    ;;
  --cpu-only)
    cpu_only=1
    shift
    ;;
  -g | --gpu-id)
    gpu_id="${2}"
    shift
    shift
    ;;
  -j | --threads)
    num_threads="${2}"
    shift
    shift
    ;;
  --pin)
    pin='on'
    shift
    ;;
  --pin-stride)
    pin_stride="${2}"
    shift
    shift
    ;;
  --pin-offset)
    pin_offset="${2}"
    shift
    shift
    ;;
  --use-gmx-mpi)
    use_gmx_mpi=1
    shift
    ;;
  --use-openmp-threads)
    use_openmp=1
    shift
    ;;
  --ld-seed)
    ld_seed="${2}"
    shift
    shift
    ;;
  --gen-seed)
    gen_seed="${2}"
    shift
    shift
    ;;
  --genion-seed)
    genion_seed="${2}"
    shift
    shift
    ;;
  esac
done

# Sanity checks and path preparation
if [ ! -f "${pdb_file}" ]; then
  echo "Error: Input PDB file does not exist: ${pdb_file}"
  exit 2
fi

SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
workdir=$(dirname "$pdb_file")
workdir=$(realpath "$workdir")
sample_id=$(basename "$pdb_file" .pdb)

# Command construction
if [ "${use_gmx_mpi}" == '1' ] && [ -x "$(command -v gmx_mpi)" ]; then
  gmx_exe="$(command -v gmx_mpi)"
elif [ -x "$(command -v gmx)" ]; then
  gmx_exe="$(command -v gmx)"
else
  echo 'Error: gmx/gmx_mpi not found in PATH'
  exit 2
fi

mdrun_args=('-v')
if [ "${use_openmp}" == '1' ] || [ "${use_gmx_mpi}" == '1' ]; then
  mdrun_args+=('-ntmpi' '1' '-ntomp' "${num_threads}")
else
  mdrun_args+=('-nt' "${num_threads}")
fi

if [ "${pin}" == 'on' ]; then
  mdrun_args+=('-pin' 'on' '-pinstride' "${pin_stride}" '-pinoffset' "${pin_offset}")
fi

if [ "${cpu_only}" == '0' ]; then
  mdrun_gpu_args=('-gpu_id' "${gpu_id}" '-nb' 'gpu' '-pmefft' 'gpu' '-pme' 'gpu' '-bonded' 'gpu') # -update gpu only for production runs
else
  mdrun_gpu_args=('-nb' 'cpu' '-pmefft' 'cpu' '-pme' 'cpu' '-bonded' 'cpu' '-update' 'cpu')
fi

# Introduce Amber ff19sb force field
if [ ! -f "${workdir}/${sample_id}.gro" ]; then
  mkdir -p "${workdir}/prepare"
  if [ ! -f "${workdir}/prepare/${sample_id}.gro" ]; then
    cp "${pdb_file}" "${workdir}/prepare/before_input.pdb"

    cd "${workdir}/prepare" || exit
    cp "${SCRIPT_PATH}/leap.in" "${workdir}/prepare/leap.in"
    if [ "${fix_pdb}" == '1' ]; then
      pdbfixer "${workdir}/prepare/before_input.pdb" \
        --output="${workdir}/prepare/before_input_fixed.pdb" \
        --add-atoms=none \
        --keep-heterogens=none \
        --verbose 2>&1 | tee "${workdir}/prepare/pdbfixer.log"
      "${SCRIPT_PATH}/rosetta2amber.sh" "${workdir}/prepare/before_input_fixed.pdb"
    else
      "${SCRIPT_PATH}/rosetta2amber.sh" "${workdir}/prepare/before_input.pdb"
    fi
    tleap -s -f leap.in
    python "${SCRIPT_PATH}/amber2gmx.py" 2>&1 | tee "${workdir}/prepare/amber2gmx.log"
  fi
  cp "${workdir}/prepare/gromacs.gro" "${workdir}/${sample_id}.gro"
  cp "${workdir}/prepare/gromacs.top" "${workdir}/${sample_id}.top"

  # Include solvent topology (tip3p.itp, ions_cl.itp, and ions_na.itp)
  cp "${SCRIPT_PATH}/gmx_itp/"*.itp "${workdir}/"
  sed -i '0,/moleculetype/{s/\[ moleculetype \]/; Include solvent topology\n\#include "tip3p.itp"\n\#include "ions_cl.itp"\n\#include "ions_na.itp"\n\n\[ moleculetype \]/}' "${workdir}/${sample_id}.top"
  sed -i "/cmaptypes/i\Na+           11  22.9900    1.00000     A      0.2439285      0.3658460 ;     1.37  0.0874\nCl-           17  35.45300   -1.000      A       0.441724      0.492833\nOW             6  15.999400  -0.834      A      0.3150580      0.6363860 ;     TIP3p\nHW             1   1.008000   0.417      A            0.0            0.0 ;     TIP3p\n " "${workdir}/${sample_id}.top"

  cd "${workdir}" || exit
fi

# Add water
if [ ! -f "${workdir}/solvate_${sample_id}.gro" ]; then
  "${gmx_exe}" editconf \
    -f "${workdir}/${sample_id}.gro" \
    -o "${workdir}/box_${sample_id}.gro" \
    -bt 'dodecahedron' \
    -d 0.9
  "${gmx_exe}" solvate \
    -cp "${workdir}/box_${sample_id}.gro" \
    -cs 'spc216.gro' \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/solvate_${sample_id}.gro"
fi

# Add salt
if [ ! -f "${workdir}/ions_${sample_id}.gro" ]; then
  cp "${SCRIPT_PATH}/gmx_mdp/ion.mdp" "${workdir}/"
  echo "ld-seed = ${ld_seed}" >>"${workdir}/ion.mdp"
  "${gmx_exe}" grompp \
    -f "${workdir}/ion.mdp" \
    -c "${workdir}/solvate_${sample_id}.gro" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/ions_${sample_id}.tpr"

  echo 'SOL' | "${gmx_exe}" genion \
    -s "${workdir}/ions_${sample_id}.tpr" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/ions_${sample_id}.gro" \
    -pname NA \
    -nname CL \
    -neutral \
    -seed "${genion_seed}"

  sed -i 's/ NA/Na+/g' "${workdir}/ions_${sample_id}.gro"
  sed -i 's/ CL/Cl-/g' "${workdir}/ions_${sample_id}.gro"
  echo q | "${gmx_exe}" make_ndx \
    -f "${workdir}/ions_${sample_id}.gro" \
    -o "${workdir}/index.ndx"
fi

# em
if [ ! -f "${workdir}/em_${sample_id}.gro" ]; then
  cp "${SCRIPT_PATH}/gmx_mdp/minim.mdp" "${workdir}/"
  sed -i -E "s/^(ld-seed[[:space:]]*=)[^;]*/\1 ${ld_seed}/" "${workdir}/minim.mdp"
  "${gmx_exe}" grompp \
    -f "${workdir}/minim.mdp" \
    -c "${workdir}/ions_${sample_id}.gro" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/em_${sample_id}.tpr"
  "${gmx_exe}" mdrun -deffnm "em_${sample_id}" "${mdrun_args[@]}" \
    2>&1 | tee "${workdir}/em_${sample_id}.full.log"
fi

# cg
if [ ! -f "${workdir}/cg_${sample_id}.gro" ]; then
  cp "${SCRIPT_PATH}/gmx_mdp/cg.mdp" "${workdir}/"
  sed -i -E "s/^(ld-seed[[:space:]]*=)[^;]*/\1 ${ld_seed}/" "${workdir}/cg.mdp"
  "${gmx_exe}" grompp \
    -f "${workdir}/cg.mdp" \
    -c "${workdir}/em_${sample_id}.gro" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/cg_${sample_id}.tpr"
  "${gmx_exe}" mdrun -deffnm "cg_${sample_id}" "${mdrun_args[@]}" \
    2>&1 | tee "${workdir}/cg_${sample_id}.full.log"
fi

# NVT
if [ ! -f "${workdir}/nvt_${sample_id}.gro" ]; then
  cp "${SCRIPT_PATH}/gmx_mdp/nvt.mdp" "${workdir}/"
  sed -i 's/= -DPOSRES/=-DFLEXIBLE/g' "${workdir}/nvt.mdp"
  sed -i -E "s/^(ld-seed[[:space:]]*=)[^;]*/\1 ${ld_seed}/" "${workdir}/nvt.mdp"
  sed -i -E "s/^(gen-seed[[:space:]]*=)[^;]*/\1 ${gen_seed}/" "${workdir}/nvt.mdp"
  "${gmx_exe}" grompp \
    -f "${workdir}/nvt.mdp" \
    -c "${workdir}/cg_${sample_id}.gro" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/nvt_${sample_id}.tpr"
  "${gmx_exe}" mdrun -deffnm "nvt_${sample_id}" "${mdrun_args[@]}" "${mdrun_gpu_args[@]}" \
    2>&1 | tee "${workdir}/nvt_${sample_id}.full.log"
fi

# NPT
if [ ! -f "${workdir}/npt_${sample_id}.gro" ]; then
  cp "${SCRIPT_PATH}/gmx_mdp/npt.mdp" "${workdir}/"
  sed -i 's/= -DPOSRES/=-DFLEXIBLE/g' "${workdir}/npt.mdp"
  sed -i -E "s/^(ld-seed[[:space:]]*=)[^;]*/\1 ${ld_seed}/" "${workdir}/npt.mdp"
  "${gmx_exe}" grompp \
    -f "${workdir}/npt.mdp" \
    -c "${workdir}/nvt_${sample_id}.gro" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/npt_${sample_id}.tpr"
  "${gmx_exe}" mdrun -deffnm "npt_${sample_id}" "${mdrun_args[@]}" "${mdrun_gpu_args[@]}" \
    2>&1 | tee "${workdir}/npt_${sample_id}.full.log"
fi

# Prepare production tpr
if [ ! -f "${workdir}/production_${sample_id}.tpr" ]; then
  cp "${SCRIPT_PATH}/gmx_mdp/production.mdp" "${workdir}/"
  nsteps="$(python -c "print(round($simulation_ns / 2e-6))")"
  sed -i -E "s/^(nsteps[[:space:]]*=)[^;]*.*/\1 ${nsteps};/" "${workdir}/production.mdp"
  sed -i -E "s/^(ld-seed[[:space:]]*=)[^;]*/\1 ${ld_seed}/" "${workdir}/production.mdp"
  "${gmx_exe}" grompp \
    -f "${workdir}/production.mdp" \
    -c "${workdir}/npt_${sample_id}.gro" \
    -p "${workdir}/${sample_id}.top" \
    -o "${workdir}/production_${sample_id}.tpr"
fi
