#!/usr/bin/env bash
#SBATCH -J heat_events
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o heat_events_%j.out
#SBATCH -e heat_events_%j.err

# ============================================================
# run_tempestextremes_heat_events.sh
#
# Identify spatially connected and persistent heat-wave events
# from daily binary masks created using the local p95 threshold.
#
# Input mask:
#   hot_day = 1 where tmax > local climatological p95 threshold
#             0 otherwise
#
# Event criteria:
#   - minimum spatial size: 12,500 grid cells
#   - minimum duration: 3 consecutive days
#   - diagonal connectivity enabled
#   - minimum overlap with previous and next day: 5 grid cells
#
# Requirements:
#   - TempestExtremes
#   - NetCDF
#   - Slurm
# ============================================================

set -euo pipefail


# ============================================================
# Environment
# ============================================================

module load netcdf-c

# Load TempestExtremes here if required on your system.
# module load tempestextremes


# ============================================================
# User settings
# ============================================================

DATA_ROOT="./netcdf"

START_YEAR=1980
END_YEAR=2059

VARIABLE="tmax"
THRESHOLD="p95"

MASK_VARIABLE="hot_day"
TRACK_VARIABLE="track"

MINIMUM_SIZE=12500
MINIMUM_DURATION=3

MINIMUM_OVERLAP_PREVIOUS=5
MINIMUM_OVERLAP_NEXT=5

SCENARIO="ssp585"
GRID="VIC4"


SIMULATIONS=(
    "CMIP6"
    "SRGAN_Daymet"
    "SRCNN_Daymet"
    "RegCM"
    "RegCM_Livneh"
    "RegCM_Daymet"
    "DBCCA_Livneh"
    "DBCCA_Daymet"
    "LOCA_Livneh"
)


ESMS=(
    "ACCESS-CM2"
    "BCC-CSM2-MR"
    "CNRM-ESM2-1"
    "MPI-ESM1-2-HR"
    "MRI-ESM2-0"
    "NorESM2-MM"
)


OBSERVATIONS=(
    "DaymetV4"
    "Livneh"
)


# ============================================================
# Check TempestExtremes executables
# ============================================================

if ! command -v StitchBlobs >/dev/null 2>&1
then
    echo "ERROR: StitchBlobs was not found in PATH."
    exit 1
fi

if ! command -v BlobStats >/dev/null 2>&1
then
    echo "ERROR: BlobStats was not found in PATH."
    exit 1
fi


# ============================================================
# Run TempestExtremes for one daily-mask file
# ============================================================

run_tempestextremes()
{
    local input_file="$1"
    local track_file="$2"
    local stats_file="$3"

    if [[ ! -f "${input_file}" ]]
    then
        echo "ERROR: Missing input file:"
        echo "       ${input_file}"
        exit 1
    fi

    echo "  Input mask:"
    echo "    ${input_file}"

    # --------------------------------------------------------
    # Identify and track connected heat-event objects
    # --------------------------------------------------------

    srun -n 1 StitchBlobs \
        --in "${input_file}" \
        --out "${track_file}" \
        --var "${MASK_VARIABLE}" \
        --outvar "${TRACK_VARIABLE}" \
        --diag_connect \
        --regional \
        --minsize "${MINIMUM_SIZE}" \
        --mintime "${MINIMUM_DURATION}" \
        --min_overlap_prev "${MINIMUM_OVERLAP_PREVIOUS}" \
        --min_overlap_next "${MINIMUM_OVERLAP_NEXT}" \
        --lonname lon \
        --latname lat

    # --------------------------------------------------------
    # Calculate event statistics
    # --------------------------------------------------------

    srun -n 1 BlobStats \
        --in_file "${track_file}" \
        --out_file "${stats_file}" \
        --var "${TRACK_VARIABLE}" \
        --regional \
        --out minlat,maxlat,minlon,maxlon,meanlon,meanlat,centlon,centlat,area \
        --out_headers

    echo "  Created track file:"
    echo "    ${track_file}"

    echo "  Created statistics file:"
    echo "    ${stats_file}"
}


# ============================================================
# Process ESM-driven and downscaled datasets
# ============================================================

for YEAR in $(seq "${START_YEAR}" "${END_YEAR}")
do
    echo "=================================================="
    echo "Processing heat events for year ${YEAR}"

    for ESM in "${ESMS[@]}"
    do
        if [[ "${ESM}" == "CNRM-ESM2-1" ]]
        then
            ENSEMBLE="r1i1p1f2"
        else
            ENSEMBLE="r1i1p1f1"
        fi

        for SIMULATION in "${SIMULATIONS[@]}"
        do
            DIRECTORY="${DATA_ROOT}/${SIMULATION}"

            mkdir -p "${DIRECTORY}"

            PREFIX="${ESM}_${SCENARIO}_${ENSEMBLE}_${SIMULATION}_${GRID}_${VARIABLE}_${YEAR}_${THRESHOLD}"

            INPUT_FILE="${DIRECTORY}/${PREFIX}_daily_mask.nc"

            TRACK_FILE="${DIRECTORY}/${PREFIX}_track_${MINIMUM_DURATION}days_${MINIMUM_SIZE}.nc"

            STATS_FILE="${DIRECTORY}/${PREFIX}_track_${MINIMUM_DURATION}days_${MINIMUM_SIZE}.txt"

            echo
            echo "Simulation: ${SIMULATION}"
            echo "ESM:        ${ESM}"
            echo "Year:       ${YEAR}"

            run_tempestextremes \
                "${INPUT_FILE}" \
                "${TRACK_FILE}" \
                "${STATS_FILE}"
        done
    done
done


# ============================================================
# Process Daymet v4 and Livneh v2
# ============================================================

for YEAR in $(seq "${START_YEAR}" "${END_YEAR}")
do
    for OBSERVATION in "${OBSERVATIONS[@]}"
    do
        DIRECTORY="${DATA_ROOT}/${OBSERVATION}"

        mkdir -p "${DIRECTORY}"

        PREFIX="${OBSERVATION}_${VARIABLE}_${YEAR}_${THRESHOLD}"

        INPUT_FILE="${DIRECTORY}/${PREFIX}_daily_mask.nc"

        TRACK_FILE="${DIRECTORY}/${PREFIX}_track_${MINIMUM_DURATION}days_${MINIMUM_SIZE}.nc"

        STATS_FILE="${DIRECTORY}/${PREFIX}_track_${MINIMUM_DURATION}days_${MINIMUM_SIZE}.txt"

        echo
        echo "=================================================="
        echo "Observation: ${OBSERVATION}"
        echo "Year:        ${YEAR}"

        run_tempestextremes \
            "${INPUT_FILE}" \
            "${TRACK_FILE}" \
            "${STATS_FILE}"
    done
done


echo "=================================================="
echo "All heat-event tracking calculations completed."
