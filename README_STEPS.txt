Groundwater withdrawal prediction pipeline
Python code steps

1.  Create folder using - createfolders and subfolders.py
2.  Create a folder manually to save raster files
2.  Download data sets from GEE using file(shp_files/ gmd4_boundary_wgs84.shp) - using downloadFrom_gee.py
3.  Create folders manually to save subset shapefiles ( a folder for gwl and gwl change,  a folder for quifer thickness shp files)
4.  Subset yearly data from geodatabase 
     4.1  Groundwater level & groundwater level change from file name (raw_data/nw_ks_hpa_wells.gdb)-   subset_yrly_gwl_gwlc.py
	 Interpolate using kriging
	 4.1.1 Groundwater level and groundwater level data using - interpolate_gwl_gwlc_from_shpfiles.py 
	 
	 4.2  Aquifer thickness from file name (raw_data/gmd4_aquiferThickness_deg.shp) -   subset_aquiferthickness.py
	 Interpolate using kriging
     4.2.1  Aquifer thickness - interpolate_aquiferthickness_from_shpfiles.py
	 
Create a folder to save projected raster files
5. Reproject raster files to NAD 1983 UTM Zone 14N using file(ref_rasters/reproject_ref.tif) - reproject_rasters.py
6. Create a folder to save  yearly csv file 
7. Extract raster files using(shp_files/unique_well_id_grid.shp) - extract_rasters_values_at_points.py
Note - Save all the excel files in one folder
8.  Concatenate yearly  csv files  using -  concat_csv_files.py
9. Subset groundwater withdrawals to yealy data from name(raw_data/groundwater_withdrawals.gdb) - groundwater_withdrawals.gdb
 using - subset_withdrawals.py ( save pump_all.csv to concatenated csv files folder)
10. Copy the crop water demand csv file (raw_data/cropwater_demand_all.csv) to the folder where the concat_csv_files saved
11. Merge the concatenated individual variables in to one csv file using - merge_csv_files.py - this is the final csv ingested to ml analysis
Note that change field name (csv fil created using step 11)cellid to gridid ( the aggregate_pointValues_ingrid has a cellid field and willnot aggregate values) 
12.Run machine learning analysis - randomForest_spatial_holdout_model.py
input files (ref_rasters/ref_raster_grid.tif, shp_files/ref_grid.shp)
Note - the following Python files run with in the main Ml Python codes - randomForest_spatial_holdout_model.py
   1. rasterToDataframe.py
   2. aggregate_pointValues_ingrid.py
   3. plotting.py