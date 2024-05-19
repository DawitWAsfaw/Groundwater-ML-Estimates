# Extract gdb to year shapefile
import geopandas as gpd
import numpy as np

#=============================================================================================================


def subset_withdrawals(input_dir, output_dir,shp_grid_file,driver='FileGBD',layer =0):
    
    gdb = gpd.read_file(input_dir)
    shp_grid =gpd.read_file(shp_grid_file)
    shp_grid_pro = shp_grid.set_crs(epsg='26914' ,allow_override=True)
    
    year_list = list(range(2008,2021))
    
    for kk, year in enumerate(year_list):
        pump_attr = 'AF_USED_IRR_' + str(year)
        pump_acre = 'ACRES_' + str(year)
        acf_to_mm = 4047000000
        pump_acre_zero = gdb[gdb[pump_acre] == 0].index
        gdb.drop(pump_acre_zero , inplace=True)
        if kk==0:
            new_shp = gdb[['PDIV_ID','long_nad83', 'lat_nad83','geometry']]
            new_shp['year']= year    
            new_shp['pump_aft_acre'] = gdb[pump_attr]/gdb[pump_acre]
            new_shp['pump_mm'] = new_shp['pump_aft_acre'] * 304.8
            new_shp['irr_area_mm2'] = gdb[pump_acre]  * acf_to_mm # better to convert value to mm3 in other way
            new_shp['pump_mm3'] =  new_shp['pump_mm'] * new_shp['irr_area_mm2']
        else:
            new_shp2= gdb[['PDIV_ID','long_nad83', 'lat_nad83','geometry']]
            new_shp2['year']= year
            new_shp2['pump_aft_acre'] = gdb[pump_attr]/gdb[pump_acre]
            new_shp2['irr_area_mm2'] =gdb[pump_acre] * acf_to_mm
            
            new_shp2['pump_mm'] = new_shp2['pump_aft_acre'] * 304.8
            new_shp2['pump_mm3'] = new_shp2['pump_mm'] * new_shp2['irr_area_mm2']
            new_shp = new_shp.append(new_shp2)
            
    # new_shp.to_file(output_dir + 'pump_all.shp')
#============================================================================================================= 
    # adding cellId to pumping data for spatial holdout
    shp_grid_pro = shp_grid.set_crs(epsg='26914' ,allow_override=True)
    joined = gpd.sjoin(new_shp, shp_grid_pro,how ='left')
    # joined.to_file(output_dir + 'pump_all_id_grid.shp')
    joined = joined.drop('geometry',axis=1)
    
#=============================================================================================================    
    pump_all= joined
    pump_all = pump_all[pump_all.pump_mm > 0]
    pump_all = pump_all.rename(columns={'cellid':'gridid' }, errors="raise")

    pump_all = pump_all.drop(['index_right','row','col'],axis=1)
    pump_all.to_csv(output_dir + 'pump_all.csv',index = False)   
    
 


                        
            
            
            