;=======================================================================
; February 2021 CONUS cold-wave temperature-anomaly map
;
; Purpose
; -------
; This NCL script creates the spatial map of minimum-temperature anomaly
; during the February 2021 cold wave over the contiguous United States
; (CONUS).
;
; The anomaly is calculated as:
;
;     February 2021 event-period mean tmin
;       minus
;     2016–2022 climatological mean tmin for the same calendar days
;
; Grid cells outside the EIA balancing-authority region mask are excluded.
;
; Output
; ------
; Daymet_tmincw_Feb2021_anom.pdf
;
; Data requirements
; -----------------
; 1. DaymetV4_VIC4_tmin_2021.nc
; 2. DaymetV4_VIC4_tmin_2016-2022.nc
; 3. DaymetV4_VIC4_prcp_1980.nc
;       Used only to obtain latitude and longitude coordinates.
; 4. US_Electric_Grid_Monitor_BA_Region.nc
;       Provides the EIA-region mask.
;
;
; Software
; --------
; NCAR Command Language (NCL)
;
;
;=======================================================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin

  ;---------------------------------------------------------------------
  ; User configuration
  ;---------------------------------------------------------------------

  OBSERVATION = "DaymetV4"
  YEAR        = 2021

  DATA_ROOT = "./Data"

  TMIN_DIR  = DATA_ROOT + "/" + OBSERVATION + "/tmin/"
  PRCP_FILE = DATA_ROOT + "/DaymetV4/prcp/DaymetV4_VIC4_prcp_1980.nc"

  MASK_FILE = "US_Electric_Grid_Monitor_BA_Region.nc"

  EVENT_FILE = TMIN_DIR + OBSERVATION + "_VIC4_tmin_" + YEAR + ".nc"
  CLIM_FILE  = TMIN_DIR + OBSERVATION + "_VIC4_tmin_2016-2022.nc"

  OUTPUT_NAME = "Daymet_tmincw_Feb2021_anom"

  ; Event-period indices in the annual tmin files.
  EVENT_START_INDEX = 36
  EVENT_END_INDEX   = 58

  MISSING_VALUE = -999.0

  ;---------------------------------------------------------------------
  ; Read the EIA-region mask
  ;---------------------------------------------------------------------

  mask_file = addfile(MASK_FILE, "r")
  eia_mask  = mask_file->regions(:, :)

  ;---------------------------------------------------------------------
  ; Read February 2021 minimum temperature and climatology
  ;---------------------------------------------------------------------

  event_nc = addfile(EVENT_FILE, "r")
  clim_nc  = addfile(CLIM_FILE, "r")

  tmin_event = event_nc->tmin( \
      EVENT_START_INDEX:EVENT_END_INDEX, :, : \
  )

  tmin_clim = clim_nc->tmin( \
      EVENT_START_INDEX:EVENT_END_INDEX, :, : \
  )

  ; Standardize missing values before temporal averaging.
  tmin_event@_FillValue = MISSING_VALUE
  replace_ieeenan(tmin_event, tmin_event@_FillValue, 0)

  tmin_clim@_FillValue = MISSING_VALUE
  replace_ieeenan(tmin_clim, tmin_clim@_FillValue, 0)

  ;---------------------------------------------------------------------
  ; Read latitude and longitude coordinates
  ;---------------------------------------------------------------------

  coord_nc = addfile(PRCP_FILE, "r")
  lat      = coord_nc->lat
  lon      = coord_nc->lon

  ;---------------------------------------------------------------------
  ; Calculate the cold-wave temperature anomaly
  ;---------------------------------------------------------------------

  event_mean = dim_avg_n_Wrap(tmin_event, 0)
  clim_mean  = dim_avg_n_Wrap(tmin_clim, 0)

  tmin_anomaly = event_mean - clim_mean

  ; Attach coordinate metadata needed by the contour-map routine.
  tmin_anomaly!0   = "lat"
  tmin_anomaly!1   = "lon"
  tmin_anomaly&lat = lat
  tmin_anomaly&lon = lon

  ; Retain only grid cells inside the EIA-region mask.
  tmin_anomaly = where( \
      eia_mask.gt.0, \
      tmin_anomaly, \
      tmin_anomaly@_FillValue \
  )

  ;---------------------------------------------------------------------
  ; Create the PDF map
  ;---------------------------------------------------------------------

  wks = gsn_open_wks("pdf", OUTPUT_NAME)

  setvalues NhlGetWorkspaceObjectId()
    "wsMaximumSize" : 300000000
  end setvalues

  gsn_define_colormap(wks, "BlRe")

  res = True

  ; Drawing control
  res@gsnDraw  = False
  res@gsnFrame = False

  ; Contour settings
  res@cnFillOn           = True
  res@cnRasterModeOn     = True
  res@cnLinesOn          = False
  res@cnLineLabelsOn     = False
  res@cnInfoLabelOn      = False
  res@cnConstFEnableFill = True
  res@gsnSpreadColors    = True

  ; Anomaly range and interval
  res@cnLevelSelectionMode = "ManualLevels"
  res@cnMinLevelValF       = -10.0
  res@cnMaxLevelValF       =  10.0
  res@cnLevelSpacingF      =   1.0

  ; Color-bar settings
  res@lbLabelStride = 2

  ; Map settings
  res@gsnAddCyclic          = False
  res@mpLimitMode           = "Corners"
  res@mpLeftCornerLatF      = lat(0)
  res@mpLeftCornerLonF      = lon(0)
  res@mpRightCornerLatF     = lat(dimsizes(lat)-1)
  res@mpRightCornerLonF     = lon(dimsizes(lon)-1)
  res@mpPerimOn             = True
  res@mpLandFillColor       = "white"
  res@mpOutlineBoundarySets = "National"
  res@mpNationalLineColor   = "Black"

  ; Tick marks and title
  res@pmTickMarkDisplayMode = "Always"
  res@tiMainString          = "February 2021 minimum-temperature anomaly"
  res@tiMainFontHeightF     = 0.020
  res@tiMainOffsetYF        = 0.0

  plot = gsn_csm_contour_map(wks, tmin_anomaly, res)

  draw(plot)
  frame(wks)

end
