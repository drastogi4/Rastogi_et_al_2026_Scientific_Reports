;=======================================================================
; create_tmin_p05_daily_mask_simple.ncl
;
; Create a daily binary mask where:
;
;   cold_day = 1 when tmin < climatological p05 threshold
;   cold_day = 0 otherwise
;
; Cold season:
;   November-December of YEAR
;   January-March of YEAR+1
;
; This  script assumes each annual input file contains
; 365 daily time steps.
;=======================================================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin

;=======================================================================
; User settings
;=======================================================================

  simulations = (/ \
      "CMIP6",          \
      "SRGAN_Daymet",   \
      "SRCNN_Daymet",   \
      "RegCM",          \
      "RegCM_Livneh",   \
      "RegCM_Daymet",   \
      "DBCCA_Livneh",   \
      "DBCCA_Daymet"    \
  /)

  esms = (/ \
      "ACCESS-CM2",     \
      "BCC-CSM2-MR",   \
      "CNRM-ESM2-1",   \
      "MPI-ESM1-2-HR", \
      "MRI-ESM2-0",    \
      "NorESM2-MM"      \
  /)

  observations = (/ \
      "DaymetV4", \
      "Livneh"    \
  /)

  variable         = "tmin"
  percentile_label = "p05"
  scenario         = "ssp585"
  grid             = "VIC4"

  threshold_start_year = 1980
  threshold_end_year   = 2018

  start_year = 1980
  end_year   = 2058

  data_root      = "./data"
  threshold_root = "./netcdf"
  output_root    = "./netcdf"

  mask_file = "./NCDC_regions_new.nc"

  daymet_directory = data_root + "/DaymetV4/tmin"
  livneh_directory = data_root + "/Livneh/tmin"

  daymet_prefix = "DaymetV4_tmin"
  livneh_prefix = "Livneh_tmin"


;=======================================================================
; Read analysis-domain mask
;=======================================================================

  fmask = addfile(mask_file, "r")

  ncei = fmask->ncei(0,:,:)

  ncei = where( \
      ncei.eq.99, \
      ncei@_FillValue, \
      ncei \
  )


;=======================================================================
; Process ESM-driven datasets
;=======================================================================

  do ss = 0, dimsizes(simulations)-1

    simulation = simulations(ss)

    output_directory = output_root + "/" + simulation + "/"

    system("mkdir -p " + output_directory)

    do ee = 0, dimsizes(esms)-1

      esm = esms(ee)

      print("==================================================")
      print("Simulation: " + simulation)
      print("ESM:        " + esm)

      if (esm.eq."CNRM-ESM2-1") then
        ensemble = "r1i1p1f2"
      else
        ensemble = "r1i1p1f1"
      end if

      input_directory = data_root + "/" + \
                        esm + "_" + scenario + "_" + ensemble + "_" + \
                        simulation + "/" + variable + "/"

      file_prefix = esm + "_" + scenario + "_" + ensemble + "_" + \
                    simulation + "_" + grid + "_" + variable

;-----------------------------------------------------------------------
; Read climatological p05 threshold
;-----------------------------------------------------------------------

      threshold_file = threshold_root + "/" + simulation + "/" + \
                       file_prefix + "_" + threshold_start_year + "-" + \
                       threshold_end_year + "_" + percentile_label + ".nc"

      if (.not.isfilepresent(threshold_file)) then
        print("FATAL: Missing threshold file:")
        print(threshold_file)
        exit
      end if

      fthreshold = addfile(threshold_file, "r")

      threshold_all = fthreshold->$variable$

      threshold_dimensions = dimsizes(threshold_all)

      if (dimsizes(threshold_dimensions).eq.3) then
        threshold = threshold_all(0,:,:)
      else
        threshold = threshold_all
      end if

;-----------------------------------------------------------------------
; Process each November-March season
;-----------------------------------------------------------------------

      years = ispan(start_year, end_year, 1)

      do yy = 0, dimsizes(years)-1

        year1 = years(yy)
        year2 = year1 + 1

        print("Processing season: " + year1 + "-" + year2)

        input_file1 = input_directory + file_prefix + "_" + year1 + ".nc"
        input_file2 = input_directory + file_prefix + "_" + year2 + ".nc"

        if (.not.isfilepresent(input_file1)) then
          print("FATAL: Missing input file:")
          print(input_file1)
          exit
        end if

        if (.not.isfilepresent(input_file2)) then
          print("FATAL: Missing input file:")
          print(input_file2)
          exit
        end if

        fyear1 = addfile(input_file1, "r")
        fyear2 = addfile(input_file2, "r")

;-----------------------------------------------------------------------
; Select November-December and January-March
;
; Assumes 365-day annual files.
;-----------------------------------------------------------------------

        temperature1 = fyear1->$variable$(304:364,:,:)
        temperature2 = fyear2->$variable$(0:89,:,:)

        temperature = array_append_record( \
            temperature1, \
            temperature2, \
            0 \
        )

;-----------------------------------------------------------------------
; Apply regional mask
;-----------------------------------------------------------------------

        ncei_3d = conform( \
            temperature, \
            ncei, \
            (/1,2/) \
        )

        temperature = where( \
            ismissing(ncei_3d), \
            temperature@_FillValue, \
            temperature \
        )

;-----------------------------------------------------------------------
; Apply p05 threshold
;-----------------------------------------------------------------------

        threshold_3d = conform( \
            temperature, \
            threshold, \
            (/1,2/) \
        )

        cold_day = where( \
            temperature.lt.threshold_3d, \
            1, \
            0 \
        )

        copy_VarCoords(temperature, cold_day)

        cold_day@long_name = \
            "Daily minimum temperature below local p05 threshold"

        cold_day@description = \
            "1 where tmin is below the 1980-2018 climatological p05 threshold; 0 otherwise"

        cold_day@units = "1"
        cold_day@threshold_reference_period = "1980-2018"
        cold_day@season = "November-March"

;-----------------------------------------------------------------------
; Write output
;-----------------------------------------------------------------------

        output_file = output_directory + file_prefix + "_" + year1 + \
                      "_" + percentile_label + "_daily_mask.nc"

        system("rm -f " + output_file)

        fout = addfile(output_file, "c")

        fout->cold_day = cold_day

        print("Created:")
        print(output_file)

;-----------------------------------------------------------------------
; Clean temporary variables
;-----------------------------------------------------------------------

        delete(fyear1)
        delete(fyear2)
        delete(temperature1)
        delete(temperature2)
        delete(temperature)
        delete(ncei_3d)
        delete(threshold_3d)
        delete(cold_day)

      end do

      delete(fthreshold)
      delete(threshold_all)
      delete(threshold)
      delete(threshold_dimensions)

    end do
  end do


;=======================================================================
; Process Daymet and Livneh
;=======================================================================

  do oo = 0, dimsizes(observations)-1

    observation = observations(oo)

    if (observation.eq."DaymetV4") then

      input_directory   = daymet_directory + "/"
      observation_prefix = daymet_prefix

    else

      input_directory   = livneh_directory + "/"
      observation_prefix = livneh_prefix

    end if

    output_directory = output_root + "/" + observation + "/"

    system("mkdir -p " + output_directory)

    threshold_file = threshold_root + "/" + observation + "/" + \
                     observation_prefix + "_" + \
                     threshold_start_year + "-" + \
                     threshold_end_year + "_" + \
                     percentile_label + ".nc"

    if (.not.isfilepresent(threshold_file)) then
      print("FATAL: Missing observational threshold file:")
      print(threshold_file)
      exit
    end if

    fthreshold = addfile(threshold_file, "r")

    threshold_all = fthreshold->$variable$

    threshold_dimensions = dimsizes(threshold_all)

    if (dimsizes(threshold_dimensions).eq.3) then
      threshold = threshold_all(0,:,:)
    else
      threshold = threshold_all
    end if

    years = ispan(start_year, end_year, 1)

    do yy = 0, dimsizes(years)-1

      year1 = years(yy)
      year2 = year1 + 1

      print("==================================================")
      print("Observation: " + observation)
      print("Season:      " + year1 + "-" + year2)

      input_file1 = input_directory + observation_prefix + "_" + \
                    year1 + ".nc"

      input_file2 = input_directory + observation_prefix + "_" + \
                    year2 + ".nc"

      if (.not.isfilepresent(input_file1)) then
        print("FATAL: Missing observational input file:")
        print(input_file1)
        exit
      end if

      if (.not.isfilepresent(input_file2)) then
        print("FATAL: Missing observational input file:")
        print(input_file2)
        exit
      end if

      fyear1 = addfile(input_file1, "r")
      fyear2 = addfile(input_file2, "r")

      temperature1 = fyear1->$variable$(304:364,:,:)
      temperature2 = fyear2->$variable$(0:89,:,:)

      temperature = array_append_record( \
          temperature1, \
          temperature2, \
          0 \
      )

      ncei_3d = conform( \
          temperature, \
          ncei, \
          (/1,2/) \
      )

      temperature = where( \
          ismissing(ncei_3d), \
          temperature@_FillValue, \
          temperature \
      )

      threshold_3d = conform( \
          temperature, \
          threshold, \
          (/1,2/) \
      )

      cold_day = where( \
          temperature.lt.threshold_3d, \
          1, \
          0 \
      )

      copy_VarCoords(temperature, cold_day)

      cold_day@long_name = \
          "Daily minimum temperature below local p05 threshold"

      cold_day@description = \
          "1 where tmin is below the 1980-2018 climatological p05 threshold; 0 otherwise"

      cold_day@units = "1"
      cold_day@threshold_reference_period = "1980-2018"
      cold_day@season = "November-March"

      output_file = output_directory + observation_prefix + "_" + \
                    year1 + "_" + percentile_label + \
                    "_daily_mask.nc"

      system("rm -f " + output_file)

      fout = addfile(output_file, "c")

      fout->cold_day = cold_day

      print("Created:")
      print(output_file)

      delete(fyear1)
      delete(fyear2)
      delete(temperature1)
      delete(temperature2)
      delete(temperature)
      delete(ncei_3d)
      delete(threshold_3d)
      delete(cold_day)

    end do

    delete(fthreshold)
    delete(threshold_all)
    delete(threshold)
    delete(threshold_dimensions)

  end do

  print("==================================================")
  print("All cold-day masks completed.")

end
