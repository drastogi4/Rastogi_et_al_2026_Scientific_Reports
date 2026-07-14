#!/usr/bin/env bash

# ============================================================
# calculate_tmin_p05_climatology.sh
#
# Calculate annual 5th-percentile daily minimum temperature
# at each grid cell, then average the annual percentile fields
# over 1980-2018 to create a climatological threshold.
#
# The script processes:
#   1. ESM-driven downscaled simulations
#   2. Daymet v4
#   3. Livneh v2
#
# Requirements:
#   - Bash
#   - Climate Data Operators (CDO)
#
# Expected downscaled input structure:
#
#   DATA_ROOT/
#   └── <ESM>_ssp585_<ensemble>_<simulation>/
#       └── tmin/
#           └── <ESM>_ssp585_<ensemble>_<simulation>_VIC4_tmin_<year>.nc
#
# Expected observational input structure:
#
#   DAYMET_DIR/
#   └── DaymetV4_tmin_<year>.nc
#
#   LIVNEH_DIR/
#   └── Livneh_tmin_<year>.nc
#
# Usage:
#
#   bash calculate_tmin_p05_climatology.sh
#
# Paths can also be supplied through environment variables:
#
#   DATA_ROOT=/path/to/downscaled/data \
#   DAYMET_DIR=/path/to/daymet/tmin \
#   LIVNEH_DIR=/path/to/livneh/tmin \
#   OUTPUT_ROOT=./netcdf \
#   bash calculate_tmin_p05_climatology.sh
# ============================================================

set -euo pipefail


# ============================================================
# User settings
# ============================================================

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./netcdf}"

DAYMET_DIR="${DAYMET_DIR:-${DATA_ROOT}/DaymetV4/tmin}"
LIVNEH_DIR="${LIVNEH_DIR:-${DATA_ROOT}/Livneh/tmin}"

START_YEAR="${START_YEAR:-1980}"
END_YEAR="${END_YEAR:-2018}"

VARIABLE="tmin"
PERCENTILE="5"
PERCENTILE_LABEL="p05"

SCENARIO="ssp585"
GRID="VIC4"

DAYMET_PREFIX="${DAYMET_PREFIX:-DaymetV4_tmin}"
LIVNEH_PREFIX="${LIVNEH_PREFIX:-Livneh_tmin}"


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
# Function to calculate annual p05
# ============================================================

calculate_annual_p05()
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

        TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/tmin-p05.XXXXXX")"

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

            calculate_annual_p05 \
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
# Process Daymet
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

TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/daymet-p05.XXXXXX")"

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

    calculate_annual_p05 \
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

TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/livneh-p05.XXXXXX")"

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

    calculate_annual_p05 \
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
echo "All annual percentile and climatology calculations completed."
