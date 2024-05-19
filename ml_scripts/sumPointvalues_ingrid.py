# Create a function that aggregates well pumping values with in a grid
import os
import numpy as np
import rasterio as rio
from rasterio import features
import geopandas as gpd
from glob import glob

#==============================================================================================================
def burn_raster_from_shapefile(shp,rast,fname,colname,dtype='int16',nodata=0):
    meta = rast.meta
    meta['dtype'] = dtype
    meta['nodata'] = nodata
    
    with rio.open(fname, 'w+', **meta) as out:
        out_arr = out.read(1)    
        shapes = ((geom,value) for geom, value in zip(shp.geometry, shp[colname]))
        
        burned = rio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write(burned,1)
        
    return(burned)

#===============================================================================================================

def aggregate_pointValues_ingrid(input_dir,shp_grid,ref_rast, output_dir, pattern = '*.shp'):
    names = []
    shp_grid= gpd.read_file(shp_grid)
    ref_rast= rio.open(ref_rast)
    for file in glob(input_dir + pattern):
        variable = os.path.basename(file).split('/')[-1]
        name = variable[variable.rfind(os.sep) +1: variable.rfind('.')]
        names.append(name)
    names = list(set(names))
    for name in names:
        files = glob(input_dir + name +pattern)
        for kk, file in enumerate(files):
            point_shp = gpd.read_file(file)
            point_shp = point_shp.set_crs(epsg='4326',allow_override=True)
            point_shp_proj = point_shp.to_crs(epsg= '32614')
            
            
            joined = gpd.sjoin(point_shp_proj,shp_grid,how='left')
            aggregated = joined.groupby('cellid').sum()
            area_mm = 2000000*2000000
            column_name= point_shp_proj.columns[4] # columns name index should be change depending on the position of the field name
            aggregated[column_name] = aggregated[column_name]/area_mm 

            grid_shp = shp_grid.set_index('cellid').join(aggregated[column_name])
            filename =  output_dir + file[file.rfind(os.sep) + 1: file.rfind('.') + 1] + 'tif'
            print(filename)

            burned = burn_raster_from_shapefile(grid_shp,ref_rast,filename, colname=column_name,dtype='float32',nodata=-999)
            burned[np.isnan(burned)] = 0


        
        
# #==================================================================================================================




        
