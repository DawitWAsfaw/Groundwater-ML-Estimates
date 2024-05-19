import geopandas as gpd



#=============================================================================================================
def subset_yrly_gwl_gwlc(input_file,output_dir):
    year_list = list(range(2007,2021))
    gdb = gpd.read_file(input_file,driver='FileGBD',layer =0)
    print(gdb.columns)
    for kk, year in enumerate(year_list):
        gwl = "winter_1yr_" + str(year)
        gwlchange = 'chng_' + str(year) + '_' + str(year+1)
        
        if kk ==0:
            gwl_shp = gdb[['USGS_ID','LATITUDE','LONGITUDE', 'LOCAL_WELL_NUMBER','USE_OF_WATER_PRIMARY','DEPTH_OF_WELL','geometry']]
            gwlchange_shp = gdb[['USGS_ID','LATITUDE','LONGITUDE', 'LOCAL_WELL_NUMBER','USE_OF_WATER_PRIMARY','DEPTH_OF_WELL','geometry']]
            gwl_shp['year'] = year
            gwlchange_shp['year'] = year
            gwl_shp['gwl'] = gdb[gwl] * 0.3048
            gwl_shp['gwl']=gwl_shp['gwl'].round(3)
            gwlchange_shp['gwlchan'] = gdb[gwlchange] * 0.3048
            gwlchange_shp['gwlchan'] = gwlchange_shp['gwlchan'].round(3)
        else:
            gwl_shp2 = gdb[['USGS_ID','LATITUDE','LONGITUDE', 'LOCAL_WELL_NUMBER','USE_OF_WATER_PRIMARY','DEPTH_OF_WELL','geometry']]
            gwlchange_shp2 = gdb[['USGS_ID','LATITUDE','LONGITUDE', 'LOCAL_WELL_NUMBER','USE_OF_WATER_PRIMARY','DEPTH_OF_WELL','geometry']]
            gwl_shp2['year'] = year
            gwlchange_shp2['year'] = year
            gwl_shp2['gwl'] = gdb[gwl]* 0.3048
            gwl_shp['gwl']=gwl_shp['gwl'].round(3)
            gwlchange_shp2['gwlchan'] = gdb[gwlchange]* 0.3048
            gwlchange_shp['gwlchan'] = gwlchange_shp['gwlchan'].round(3)
            gwl_shp = gwl_shp.append(gwl_shp2)
            gwlchange_shp = gwlchange_shp.append(gwlchange_shp2)
            gwl_shp.to_file(output_dir + 'gwl_{}.shp'.format(year))
            gwlchange_shp.to_file(output_dir + 'gwlchange_{}.shp'.format(year))

#=============================================================================================================


input_file= 'nw_ks_hpa_wells.gdb'
output_dir = ''

subset_yrly_gwl_gwlc(input_file,output_dir)