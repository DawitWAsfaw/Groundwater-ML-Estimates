import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from pykrige.ok import OrdinaryKriging

# =============================================================================================================

def saveraster_with_transform(data,fname,transform,drivername='GTiff',epsg=4269,datatype='float32',bands=1):
    # if crs !='':
    #     crs=rio.crs.CRS.from_proj4(crs)
    if epsg!='':
        crs=rio.crs.CRS.from_epsg(epsg)
    with rio.open(
            fname,'w',driver=drivername,
            height=data.shape[0],
            width=data.shape[1],
            count=bands,
            dtype=datatype,
            crs=crs,
            transform=transform) as dst:
        dst.write(np.float32(data),bands)
        
        
def interpolate_gwl_gwlc_from_shpfiles(input_dir,output_dir, pattern = '*.shp'):
    names = []
    for file in glob(input_dir + pattern):
        variable = os.path.basename(file).split('/')[-1]
        name = variable[variable.rfind(os.sep) +1: variable.rfind('_') + 1]
        names.append(name)
    names = list(set(names))
    for name in names:
        files = glob(input_dir + name +pattern)
        print(files)
        for kk, file in enumerate(files):
            gdb_file = gpd.read_file(file)
            long = gdb_file['LONGITUDE'].to_numpy()
            lat = gdb_file['LATITUDE'].to_numpy()
            min_long = min(long)
            max_long = max(long)
            min_lat = min(lat)
            max_lat = max(lat)
            gridx = np.arange(min_long,max_long, 0.005,dtype='float64')
            gridy = np.arange(min_lat,max_lat, 0.005,dtype='float64')
            gdb_file =  gdb_file.dropna()
            gdb_file= gdb_file.set_crs(epsg='4326',allow_override=True)
            gdb_file = gdb_file.to_crs(epsg= '4326')
            gdb_file =  gdb_file.dropna()
               
            long = gdb_file['LONGITUDE'].to_numpy()
            lat = gdb_file['LATITUDE'].to_numpy()
            gwl_value= gdb_file[gdb_file.columns[7]].to_numpy() # Note replace gwl with column index value

            shp_krig = OrdinaryKriging(long, lat, gwl_value,variogram_model= 'exponential',
                                      verbose=True,
                                      enable_plotting=True,
                                      nlags=100)
            plt.close()
            filename =  output_dir + file[file.rfind(os.sep) + 1: file.rfind('.') + 1] + 'tif'
            shp_kriged, uncertainity = shp_krig.execute('grid', gridx,gridy,mask=False)
            transform = from_origin(min_long,max_lat, 0.005, 0.005)
            
            saveraster_with_transform(shp_kriged,filename,transform ,drivername='GTiff',epsg=4326,datatype='float32',bands=1) 
# =============================================================================================================

input_dir = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/'\
    'groundwater_withdrawal_prediciton/for_monty/trial/yearly_shpfiles/'
output_dir  = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/'\
    'Spring2024/Research/groundwater_withdrawal_prediciton/for_monty/trial/raster_files/tiff/'

interpolate_gwl_gwlc_from_shpfiles(input_dir,output_dir, pattern = '*.shp')