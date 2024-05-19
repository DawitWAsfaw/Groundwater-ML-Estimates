# Source StackExchange - https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations

import rasterio as rio
import geopandas as gpd
import pandas as pd
import os
from glob import glob
#==================================================================================================================
# Extract Raster file from a singe raster file

#==================================================================================================================
def extract_raster_values_at_points(raster_path,points_path,output_path):
    values = []
    raster = rio.open(raster_path)
    point_gdf = gpd.read_file(points_path)
    variable =os.path.basename(raster_path).split('/')[-1]
    variable = variable[variable.rfind(os.sep) + 1: variable.rfind('.')]
    variable_name =  variable[ variable.rfind(os.sep) + 1:  variable.rfind('_')]
    # print(variable_name)
    year = variable[-4:]
    df1 = point_gdf['PDIV_ID']
    for index,point in point_gdf.iterrows():
        x,y = point.geometry.x, point.geometry.y
        row,col = raster.index(x,y)  # get the row and column index of in the raster
        value = raster.read(1, window=((row, row+1), (col, col+1))) # read the raster at the point
        value = value.reshape(value.shape[0] * value .shape[1])
        values.append(value)
    
    df2 = pd.DataFrame(data ={variable_name:values})
    df2[variable_name] = df2[variable_name].str.get(0)
    df = pd.concat([df1, df2], axis=1)
    df.insert(loc =0,column ='year',value = year)
    df.to_csv(output_path + variable + '.csv',index= False)

#==================================================================================================================
# Extract Raster file from a multiple raster files

#==================================================================================================================
def extract_rasters_values_at_points(input_raster_dir,point_shp, output_dir,pattern='*.tif'):

    for file in glob(input_raster_dir + pattern):
        extract_raster_values_at_points(file,point_shp,output_path)
        

input_raster_dir = 'projected_raster_folder/'
point_shp =  'unique_well_id_grid.shp'
output_path = input_raster_dir + 'csv_yearly/'
  
extract_rasters_values_at_points(input_raster_dir,point_shp, output_path,pattern='*.tif')












