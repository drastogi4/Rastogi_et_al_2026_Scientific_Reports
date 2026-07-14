;=======================================================================
; create_tmax_p95_daily_mask.ncl
;
; Create a daily binary mask where:
;
;   hot_day = 1 when tmax > climatological p95 threshold
;   hot_day = 0 otherwise
;
; Warm season:
;   May 1 through September 30
;
; This script assumes each annual input file contains
; 365 daily time steps.
;
; Output:
;   hot_day(time, lat, lon)
;
; The output is a daily binary mask, not a seasonal count.
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
      "DBCCA_Daymet",   \
      "LOCA_Livneh"     \
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

  variable         = "tmax"
  percentile_label = "p95"
  scenario         = "ssp585"
  grid             = "VIC4"

  threshold_start_year = 1980
  threshold_end_year   = 2018

  start_year = 1980
  end_year   = 2059

  data_root      = "./data"
  threshold_root = "./netcdf"
  output_root    = "./netcdf"

  mask_file = "./NCDC_regions_new.nc"

  daymet_directory = data_root + "/DaymetV4/tmax"
  livneh_directory = data_root + "/Livneh/tmax"

  daymet_prefix = "DaymetV4_tmax"
  livneh_prefix = "Livneh_tmax"


;=======================================================================
; Read analysis-domain mask
;=======================================================================

  if (.not.isfilepresent(mask_file)) then
    print("FATAL: Missing mask file:")
    print(mask_file)
    exit
  end if

  fmask = addfile(mask_file, "r")

  ncei = fmask->ncei(0,:,:)

  ; Exclude cells assigned code 99.
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

;-----------------------------------------------------------------------
; Select ensemble member
;-----------------------------------------------------------------------

      if (esm.eq."CNRM-ESM2-1") then
        ensemble = "r1i1p1f2"
      else
        ensemble = "r1i1p1f1"
      end if

;-----------------------------------------------------------------------
; Construct input directory and filename prefix
;-----------------------------------------------------------------------

      input_directory = data_root + "/" + \
                        esm + "_" + scenario + "_" + ensemble + "_" + \
                        simulation + "/" + variable + "/"

      file_prefix = esm + "_" + scenario + "_" + ensemble + "_" + \
                    simulation + "_" + grid + "_" + variable

;-----------------------------------------------------------------------
; Read climatological p95 threshold
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

      threshold = fthreshold->$variable$(0,:,:)

;-----------------------------------------------------------------------
; Loop over years
;-----------------------------------------------------------------------

      years = ispan(start_year, end_year, 1)

      do yy = 0, dimsizes(years)-1

        year = years(yy)

        print("-----------------------------------------------")
        print("Processing year: " + year)

        input_file = input_directory + file_prefix + "_" + year + ".nc"

        if (.not.isfilepresent(input_file)) then
          print("FATAL: Missing input file:")
          print(input_file)
          exit
        end if

        fin = addfile(input_file, "r")

;-----------------------------------------------------------------------
; Select May 1 through September 30
;
; Index 120 corresponds to May 1 and index 272 corresponds to
; September 30 for a 365-day year.
;-----------------------------------------------------------------------

        temperature = fin->$variable$(120:272,:,:)

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
; Apply p95 threshold
;-----------------------------------------------------------------------

        threshold_3d = conform( \
            temperature, \
            threshold, \
            (/1,2/) \
        )

        hot_day = where( \
            temperature.gt.threshold_3d, \
            1, \
            0 \
        )

        copy_VarCoords(temperature, hot_day)

        ; Convert missing cells to zero, following the original workflow.
        hot_day = where(ismissing(hot_day), 0, hot_day)

        hot_day@long_name = \
            "Daily maximum temperature above local p95 threshold"

        hot_day@description = \
            "1 where tmax exceeds the 1980-2018 climatological p95 threshold; 0 otherwise"

        hot_day@units = "1"
        hot_day@threshold_reference_period = "1980-2018"
        hot_day@season = "May-September"

;-----------------------------------------------------------------------
; Write output
;-----------------------------------------------------------------------

        output_file = output_directory + file_prefix + "_" + year + \
                      "_" + percentile_label + "_daily_mask.nc"

        system("rm -f " + output_file)

        fout = addfile(output_file, "c")

        fout->hot_day = hot_day

        print("Created:")
        print(output_file)

;-----------------------------------------------------------------------
; Clean temporary variables
;-----------------------------------------------------------------------

        delete(fin)
        delete(temperature)
        delete(ncei_3d)
        delete(threshold_3d)
        delete(hot_day)

      end do

      delete(fthreshold)
      delete(threshold)

    end do
  end do


;=======================================================================
; Process Daymet and Livneh
;=======================================================================

  do oo = 0, dimsizes(observations)-1

    observation = observations(oo)

    if (observation.eq."DaymetV4") then

      input_directory    = daymet_directory + "/"
      observation_prefix = daymet_prefix

    else

      input_directory    = livneh_directory + "/"
      observation_prefix = livneh_prefix

    end if

    output_directory = output_root + "/" + observation + "/"

    system("mkdir -p " + output_directory)

;-----------------------------------------------------------------------
; Read observational p95 threshold
;-----------------------------------------------------------------------

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

    threshold = fthreshold->$variable$(0,:,:)

;-----------------------------------------------------------------------
; Loop over observational years
;-----------------------------------------------------------------------

    years = ispan(start_year, end_year, 1)

    do yy = 0, dimsizes(years)-1

      year = years(yy)

      print("==================================================")
      print("Observation: " + observation)
      print("Year:        " + year)

      input_file = input_directory + observation_prefix + "_" + \
                   year + ".nc"

      if (.not.isfilepresent(input_file)) then
        print("FATAL: Missing observational input file:")
        print(input_file)
        exit
      end if

      fin = addfile(input_file, "r")

;-----------------------------------------------------------------------
; Select May 1 through September 30
;-----------------------------------------------------------------------

      temperature = fin->$variable$(120:272,:,:)

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
; Apply p95 threshold
;-----------------------------------------------------------------------

      threshold_3d = conform( \
          temperature, \
          threshold, \
          (/1,2/) \
      )

      hot_day = where( \
          temperature.gt.threshold_3d, \
          1, \
          0 \
      )

      copy_VarCoords(temperature, hot_day)

      hot_day = where(ismissing(hot_day), 0, hot_day)

      hot_day@long_name = \
          "Daily maximum temperature above local p95 threshold"

      hot_day@description = \
          "1 where tmax exceeds the 1980-2018 climatological p95 threshold; 0 otherwise"

      hot_day@units = "1"
      hot_day@threshold_reference_period = "1980-2018"
      hot_day@season = "May-September"

;-----------------------------------------------------------------------
; Write observational mask
;-----------------------------------------------------------------------

      output_file = output_directory + observation_prefix + "_" + \
                    year + "_" + percentile_label + \
                    "_daily_mask.nc"

      system("rm -f " + output_file)

      fout = addfile(output_file, "c")

      fout->hot_day = hot_day

      print("Created:")
      print(output_file)

;-----------------------------------------------------------------------
; Clean temporary variables
;-----------------------------------------------------------------------

      delete(fin)
      delete(temperature)
      delete(ncei_3d)
      delete(threshold_3d)
      delete(hot_day)

    end do

    delete(fthreshold)
    delete(threshold)

  end do

  print("==================================================")
  print("All hot-day masks completed.")

end
