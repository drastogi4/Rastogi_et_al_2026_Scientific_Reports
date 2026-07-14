;=======================================================================
; June 2022 CONUS heatwave temperature-anomaly map
;
; Purpose
; -------
; This NCL script creates the spatial map of maximum-temperature anomaly
; during the June 2022 heatwave over the contiguous United States (CONUS).
;
; The anomaly is calculated as:
;
;     June 2022 event-period mean tmax
;       minus
;     2016–2021 climatological mean tmax for the same calendar days
;
; Grid cells outside the EIA balancing-authority region mask are excluded.
;
; Output
; ------
; Daymet_tmaxhw_June2022_anom.pdf
;
; Data requirements
; -----------------
; 1. DaymetV4_VIC4_tmax_2022.nc
; 2. DaymetV4_VIC4_tmax_2016-2021.nc
; 3. DaymetV4_VIC4_prcp_1980.nc
;       Used only to obtain the latitude and longitude coordinates.
; 4. US_Electric_Grid_Monitor_BA_Region_Boundary_Approximations.nc
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
  YEAR        = 2022

  DATA_ROOT = "./Data"

  TMAX_DIR  = DATA_ROOT + "/" + OBSERVATION + "/tmax/"
  PRCP_FILE = DATA_ROOT + "/DaymetV4/prcp/DaymetV4_VIC4_prcp_1980.nc"

  MASK_FILE = "US_Electric_Grid_Monitor_BA_Region.nc"

  EVENT_FILE = TMAX_DIR + OBSERVATION + "_VIC4_tmax_" + YEAR + ".nc"
  CLIM_FILE  = TMAX_DIR + OBSERVATION + "_VIC4_tmax_2016-2021.nc"

  OUTPUT_NAME = "Daymet_tmaxhw_June2022_anom"

  ; Warm-season subset used by the original analysis.
  WARM_SEASON_START = 120
  WARM_SEASON_END   = 272

  ; Heatwave-period indices relative to the warm-season subset.
  EVENT_START_INDEX = 35
  EVENT_END_INDEX   = 56

  MISSING_VALUE = -999.0

  ;---------------------------------------------------------------------
  ; Read the EIA-region mask
  ;---------------------------------------------------------------------

  mask_file = addfile(MASK_FILE, "r")
  eia_mask  = mask_file->regions(:, :)

  ;---------------------------------------------------------------------
  ; Read 2022 maximum temperature and the 2016–2021 climatology
  ;---------------------------------------------------------------------

  event_nc = addfile(EVENT_FILE, "r")
  clim_nc  = addfile(CLIM_FILE, "r")

  tmax_2022 = event_nc->tmax( \
      WARM_SEASON_START:WARM_SEASON_END, :, : \
  )

  tmax_clim = clim_nc->tmax( \
      WARM_SEASON_START:WARM_SEASON_END, :, : \
  )

  ; Standardize missing values before temporal averaging.
  tmax_2022@_FillValue = MISSING_VALUE
  replace_ieeenan(tmax_2022, tmax_2022@_FillValue, 0)

  tmax_clim@_FillValue = MISSING_VALUE
  replace_ieeenan(tmax_clim, tmax_clim@_FillValue, 0)

  ;---------------------------------------------------------------------
  ; Read latitude and longitude coordinates
  ;---------------------------------------------------------------------

  coord_nc = addfile(PRCP_FILE, "r")
  lat      = coord_nc->lat
  lon      = coord_nc->lon

  ;---------------------------------------------------------------------
  ; Calculate the event-period temperature anomaly
  ;---------------------------------------------------------------------

  event_mean = dim_avg_n_Wrap( \
      tmax_2022(EVENT_START_INDEX:EVENT_END_INDEX, :, :), \
      0 \
  )

  clim_mean = dim_avg_n_Wrap( \
      tmax_clim(EVENT_START_INDEX:EVENT_END_INDEX, :, :), \
      0 \
  )

  tmax_anomaly = event_mean - clim_mean

  ; Attach coordinate metadata needed by the contour-map routine.
  tmax_anomaly!0   = "lat"
  tmax_anomaly!1   = "lon"
  tmax_anomaly&lat = lat
  tmax_anomaly&lon = lon

  ; Retain only grid cells inside the EIA-region mask.
  tmax_anomaly = where( \
      eia_mask.gt.0, \
      tmax_anomaly, \
      tmax_anomaly@_FillValue \
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
  res@cnMinLevelValF       = -5.0
  res@cnMaxLevelValF       =  5.0
  res@cnLevelSpacingF      =  0.5

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
  res@tiMainString          = "June 2022 maximum-temperature anomaly"
  res@tiMainFontHeightF     = 0.020
  res@tiMainOffsetYF        = 0.0

  plot = gsn_csm_contour_map(wks, tmax_anomaly, res)

  draw(plot)
  frame(wks)

end
