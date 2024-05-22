
# Groundwater withdrawal prediction pipeline steps
#############################################
#### Datasets - saved in raw_data folder
1. Geodatabase - # Groundwater withdrawals - groundwater_withdrawals.gdb
                 # Aquifer thickness - gmd4_aquiferThickness_deg.shp
		 # Groundwater level and groundwater level changes - nw_ks_hpa_wells.gdb
2. Crop water demand - cropwater_demand_all.csv
3. 2km  Reference raster
4. 2km by 2km shapefile - ref_raster_grid.tif

#############################################
####   File management       
5. Folders required to create: 
Use the create folder function
 5.1 folders - with fraction holdout for testing  - For spatio temporal estimates
      Total 10folders - including subfolder names : (model, train, test,plots) 
	  train and test folder should have - subfolder with names csv, shp, tiff
   
 5.2 create  folders - temporal estimates
    Name: Train, test, model,plots( holds PD,FI, observed vs predicted (time series and raster) plots) 
    train and test folder should have - subfolder with names csv, shp, tiff
#############################################
#### Preprocessing datasets	
6. Subset groundwater withdrawal from geodabase into yearly data and calculate withdrawals dividing 
  irrigation water use by the reported area in acres feet
  create a csv non zero groundwater withdrawal file for all the observation data from 2008 - 2020 ( create csv files for individual years)

7. Using the unique withdrawal well id, create a shapefile with the lat-long to use for extraction of predictor feature raster value at well location

9.  Using the download_gee function download the following predictor variables:
      1. precipitation
      2. Evapotranspiration 
      3  Temperature maximum
      4. Temperature minumum
      5. Slope
Note: gmd4_boundary shapefile in WGS84 projection is required

9. Subset groundwater level(gwl)  and groundwater level change(gwlc) values from geodabase and convert into yearly data
 using the interporal_gwl_gwlc function interpolate and convert gwl and gwlc to raster
 
10. Subset aquifer thickness values for gmd4 and create an interpolated raster map

11. Reproject the raster files into NAD 1983 UTM Zone 14N

12. Use unique withdrawal well station point shapefile (step 2) to extract the predictor features raster values at withdrawal location. 
Note: extract values will be storage Variablename_year.csv 

13. Compile the extracted values into csv files and merge individual predictor csv file into one csv file including withdrawal csv file ( step 1 )
#############################################
#### running machine learning 
14. Run machine learning model- total two models
  1. Spatio temporal estimate - 10 models
  2. Temporal estimate  - 2 model ( run using testing sets for wet and dry years)  by manually changing the test years
  3. Provide for the ml_function the parent directory for the 
 
 csv files for 
   1. error metrics
   2. Yearly mean withdrawal observed and actual for individaul models and models
   will be saved to the base_directory provided 
