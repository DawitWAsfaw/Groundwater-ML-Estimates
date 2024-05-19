import geopandas as gpd


#=============================================================================================================


def subset_aquiferthickness(file,output_dir):
    year_list = range(2008, 2021)
    gdb_file = gpd.read_file(file)
    lat_lon = gdb_file.centroid
    # lat_lon.to_file(output_dir + 'aquifer_centroid_lat_lon.shp')
    for year in year_list:
        aquifThick = gdb_file[['TRS','satthick_2','geometry']]
        aquifThick['satthick_m'] = aquifThick['satthick_2'] * 0.3048
        aquifThick['satthick_m'] = aquifThick['satthick_m'].round(3)
        aquifThick['lon'] = aquifThick.centroid.apply(lambda p: p.x)
        aquifThick['lat'] = aquifThick.centroid.apply(lambda p: p.y)
        aquifThick.to_file(output_dir + 'aquifThick_{}.shp'.format(year))
    
#=============================================================================================================
file = 'gmd4_aquiferThickness_deg.shp'
output_dir = 'aquiferthickness_folder'
subset_aquiferthickness(file,output_dir)