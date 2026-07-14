#!/usr/bin/env bash

# ============================================================
# calculate_tmax_p95_climatology.sh
#
# Calculate the annual 95th percentile of daily maximum
# temperature at each grid cell, then average the annual
# percentile fields over 1980-2018.
#
# Requirements:
#   - Bash
#   - Climate Data Operators (CDO)
#
# Output:
#   Annual p95 files and one 1980-2018 climatological p95 file
#   for each ESM/downscaling combination and observational
#   dataset.
# ============================================================

set -euo pipefail


# ============================================================
# User settings
# ============================================================

DATA_ROOT="./data"
OUTPUT_ROOT="./netcdf"

DAYMET_DIR="${DATA_ROOT}/DaymetV4/tmax"
LIVNEH_DIR="${DATA_ROOT}/Livneh/tmax"

START_YEAR=1980
END_YEAR=2018

VARIABLE="tmax"
PERCENTILE=95
PERCENTILE_LABEL="p95"

SCENARIO="ssp585"
GRID="VIC4"

DAYMET_PREFIX="DaymetV4_tmax"
LIVNEH_PREFIX="Livneh_tmax"


SIMULATIONS=(
    "CMIP6"
    "SRGAN_Daymet"
    "SRCNN_Daymet"
    "RegCM"
    "RegCM_Livneh"
    "RegCM_Daymet"
    "DBCCA_Livneh"
    "DBCCA_Daymet"
)


ESMS=(
    "ACCESS-CM2"
    "BCC-CSM2-MR"
    "CNRM-ESM2-1"
    "MPI-ESM1-2-HR"
    "MRI-ESM2-0"
    "NorESM2-MM"
)


# ============================================================
# Check dependencies
# ============================================================

if ! command -v cdo >/dev/null 2>&1
then
    echo "ERROR: CDO was not found in PATH."
    exit 1
fi


# ============================================================
# Function to calculate annual p95
# ============================================================

calculate_annual_p95()
{
    local input_file="$1"
    local output_file="$2"
    local temporary_directory="$3"

    local selected_file
    local minimum_file
    local maximum_file

    selected_file="${temporary_directory}/selected_${VARIABLE}.nc"
    minimum_file="${temporary_directory}/minimum_${VARIABLE}.nc"
    maximum_file="${temporary_directory}/maximum_${VARIABLE}.nc"

    cdo -O \
        "selname,${VARIABLE}" \
        "${input_file}" \
        "${selected_file}"

    cdo -O \
        yearmin \
        "${selected_file}" \
        "${minimum_file}"

    cdo -O \
        yearmax \
        "${selected_file}" \
        "${maximum_file}"

    cdo -O \
        "yearpctl,${PERCENTILE}" \
        "${selected_file}" \
        "${minimum_file}" \
        "${maximum_file}" \
        "${output_file}"
}


# ============================================================
# Process ESM-driven downscaled simulations
# ============================================================

for SIMULATION in "${SIMULATIONS[@]}"
do
    OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${SIMULATION}"

    mkdir -p "${OUTPUT_DIRECTORY}"

    for ESM in "${ESMS[@]}"
    do
        echo "=================================================="
        echo "Processing ${SIMULATION} | ${ESM}"

        if [[ "${ESM}" == "CNRM-ESM2-1" ]]
        then
            ENSEMBLE="r1i1p1f2"
        else
            ENSEMBLE="r1i1p1f1"
        fi

        INPUT_DIRECTORY="${DATA_ROOT}/${ESM}_${SCENARIO}_${ENSEMBLE}_${SIMULATION}/${VARIABLE}"

        PREFIX="${ESM}_${SCENARIO}_${ENSEMBLE}_${SIMULATION}_${GRID}_${VARIABLE}"

        if [[ ! -d "${INPUT_DIRECTORY}" ]]
        then
            echo "ERROR: Input directory not found:"
            echo "       ${INPUT_DIRECTORY}"
            exit 1
        fi

        TEMPORARY_DIRECTORY="$(
            mktemp -d "${TMPDIR:-/tmp}/tmax-p95.XXXXXX"
        )"

        ANNUAL_FILES=()

        for YEAR in $(seq "${START_YEAR}" "${END_YEAR}")
        do
            INPUT_FILE="${INPUT_DIRECTORY}/${PREFIX}_${YEAR}.nc"

            OUTPUT_FILE="${OUTPUT_DIRECTORY}/${PREFIX}_${YEAR}_${PERCENTILE_LABEL}.nc"

            echo "  Processing year ${YEAR}"

            if [[ ! -f "${INPUT_FILE}" ]]
            then
                echo "ERROR: Missing input file:"
                echo "       ${INPUT_FILE}"
                rm -rf "${TEMPORARY_DIRECTORY}"
                exit 1
            fi

            calculate_annual_p95 \
                "${INPUT_FILE}" \
                "${OUTPUT_FILE}" \
                "${TEMPORARY_DIRECTORY}"

            ANNUAL_FILES+=("${OUTPUT_FILE}")
        done

        CLIMATOLOGY_FILE="${OUTPUT_DIRECTORY}/${PREFIX}_${START_YEAR}-${END_YEAR}_${PERCENTILE_LABEL}.nc"

        echo "  Creating climatology ${START_YEAR}-${END_YEAR}"

        cdo -O \
            ensmean \
            "${ANNUAL_FILES[@]}" \
            "${CLIMATOLOGY_FILE}"

        rm -rf "${TEMPORARY_DIRECTORY}"

        echo "  Created:"
        echo "  ${CLIMATOLOGY_FILE}"
    done
done


# ============================================================
# Process Daymet v4
# ============================================================

echo "=================================================="
echo "Processing Daymet v4"

DAYMET_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/DaymetV4"

mkdir -p "${DAYMET_OUTPUT_DIRECTORY}"

if [[ ! -d "${DAYMET_DIR}" ]]
then
    echo "ERROR: Daymet directory not found:"
    echo "       ${DAYMET_DIR}"
    exit 1
fi

TEMPORARY_DIRECTORY="$(
    mktemp -d "${TMPDIR:-/tmp}/daymet-p95.XXXXXX"
)"

DAYMET_ANNUAL_FILES=()

for YEAR in $(seq "${START_YEAR}" "${END_YEAR}")
do
    INPUT_FILE="${DAYMET_DIR}/${DAYMET_PREFIX}_${YEAR}.nc"

    OUTPUT_FILE="${DAYMET_OUTPUT_DIRECTORY}/${DAYMET_PREFIX}_${YEAR}_${PERCENTILE_LABEL}.nc"

    echo "  Processing Daymet year ${YEAR}"

    if [[ ! -f "${INPUT_FILE}" ]]
    then
        echo "ERROR: Missing Daymet input file:"
        echo "       ${INPUT_FILE}"
        rm -rf "${TEMPORARY_DIRECTORY}"
        exit 1
    fi

    calculate_annual_p95 \
        "${INPUT_FILE}" \
        "${OUTPUT_FILE}" \
        "${TEMPORARY_DIRECTORY}"

    DAYMET_ANNUAL_FILES+=("${OUTPUT_FILE}")
done

DAYMET_CLIMATOLOGY_FILE="${DAYMET_OUTPUT_DIRECTORY}/${DAYMET_PREFIX}_${START_YEAR}-${END_YEAR}_${PERCENTILE_LABEL}.nc"

echo "  Creating Daymet climatology ${START_YEAR}-${END_YEAR}"

cdo -O \
    ensmean \
    "${DAYMET_ANNUAL_FILES[@]}" \
    "${DAYMET_CLIMATOLOGY_FILE}"

rm -rf "${TEMPORARY_DIRECTORY}"

echo "  Created:"
echo "  ${DAYMET_CLIMATOLOGY_FILE}"


# ============================================================
# Process Livneh v2
# ============================================================

echo "=================================================="
echo "Processing Livneh v2"

LIVNEH_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/Livneh"

mkdir -p "${LIVNEH_OUTPUT_DIRECTORY}"

if [[ ! -d "${LIVNEH_DIR}" ]]
then
    echo "ERROR: Livneh directory not found:"
    echo "       ${LIVNEH_DIR}"
    exit 1
fi

TEMPORARY_DIRECTORY="$(
    mktemp -d "${TMPDIR:-/tmp}/livneh-p95.XXXXXX"
)"

LIVNEH_ANNUAL_FILES=()

for YEAR in $(seq "${START_YEAR}" "${END_YEAR}")
do
    INPUT_FILE="${LIVNEH_DIR}/${LIVNEH_PREFIX}_${YEAR}.nc"

    OUTPUT_FILE="${LIVNEH_OUTPUT_DIRECTORY}/${LIVNEH_PREFIX}_${YEAR}_${PERCENTILE_LABEL}.nc"

    echo "  Processing Livneh year ${YEAR}"

    if [[ ! -f "${INPUT_FILE}" ]]
    then
        echo "ERROR: Missing Livneh input file:"
        echo "       ${INPUT_FILE}"
        rm -rf "${TEMPORARY_DIRECTORY}"
        exit 1
    fi

    calculate_annual_p95 \
        "${INPUT_FILE}" \
        "${OUTPUT_FILE}" \
        "${TEMPORARY_DIRECTORY}"

    LIVNEH_ANNUAL_FILES+=("${OUTPUT_FILE}")
done

LIVNEH_CLIMATOLOGY_FILE="${LIVNEH_OUTPUT_DIRECTORY}/${LIVNEH_PREFIX}_${START_YEAR}-${END_YEAR}_${PERCENTILE_LABEL}.nc"

echo "  Creating Livneh climatology ${START_YEAR}-${END_YEAR}"

cdo -O \
    ensmean \
    "${LIVNEH_ANNUAL_FILES[@]}" \
    "${LIVNEH_CLIMATOLOGY_FILE}"

rm -rf "${TEMPORARY_DIRECTORY}"

echo "  Created:"
echo "  ${LIVNEH_CLIMATOLOGY_FILE}"


echo "=================================================="
echo "All annual p95 and climatological p95 calculations completed."
