
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 00:27:13 2021
Authors
 # Dawit  (dawit.asfaw@colostate.edu)
# Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import ee
import requests
import zipfile
import os
import geopandas as gpd
from glob import glob


def downloadFrom_gee(year_list, start_month, end_month, aoi, outdir):
    """
    Download MOD16 and PRISM data. MOD16 has to be divided by 10 (line 50) as its original scale is 0.1 mm/8 days.
    :param year_list: List of years in %Y format
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param aoi_shp_file: Area of interest shapefile (must be in WGS84)
    :param outdir: Download directory
    :return: None
    """

    ee.Initialize()
    mod16= ee.ImageCollection("MODIS/006/MOD16A2")
    prism= ee.ImageCollection("OREGONSTATE/PRISM/AN81m")
    srtm_data = ee.Image('USGS/SRTMGL1_003')
    srtm_elev = srtm_data.select('elevation')
    strm_slop = ee.Terrain.slope(srtm_elev)
    
    aoi_shp = gpd.read_file(aoi_shp_file)
    minx, miny, maxx, maxy = aoi_shp.geometry.total_bounds
    gee_aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    

    for year in year_list:
        start_date = ee.Date.fromYMD(year, start_month, 1)
        if end_month == 12:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)
        else:
            end_date = ee.Date.fromYMD(year, end_month + 1, 1)
        if end_month <= start_month:
            start_date = ee.Date.fromYMD(year - 1, start_month, 1)
            
        mod16Total = mod16.select('ET').filterDate(start_date, end_date).sum().divide(10).toDouble()
        prism_ppt_Total = prism.select('ppt').filterDate(start_date, end_date).sum().toDouble()
        prismtmaxTotal = prism.select('tmax').filterDate(start_date, end_date).mean().toDouble()
        prismtminTotal = prism.select('tmin').filterDate(start_date, end_date).mean().toDouble()
        
   
    
            
        mod16_url = mod16Total.getDownloadUrl({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        prism_url = prism_ppt_Total.getDownloadUrl({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        
     
    
        prism_tmax_url = prismtmaxTotal.getDownloadUrl({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        
        prism_tmin_url = prismtminTotal.getDownloadUrl({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        
        srtm_slope_url =  strm_slop .getDownloadUrl({
            'scale': 100,
            'crs': 'EPSG:4326',
            'region': gee_aoi
            })
        
        gee_vars = ['evap_','ppt_','tmin_', 'tmax_','slope_']
        gee_links = [mod16_url,prism_url,prism_tmin_url,prism_tmax_url,srtm_slope_url]
        
        for gee_var, gee_url in zip(gee_vars, gee_links):
            local_file_name = outdir + gee_var + str(year) + '.zip'
            print('Dowloading', local_file_name, '...')
            r = requests.get(gee_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)
        

year_list = range(2017, 2021)
data_start_month = 4
data_end_month = 9
aoi_shp_file = str('gmd4_boundary_wgs84.shp')
outdir  = ''
downloadFrom_gee(year_list, data_start_month, data_end_month,aoi_shp_file, outdir)



def extract_data(zip_dir, out_dir, rename_extracted_files=False):
    """
    Extract data from zip file
    :param zip_dir: Input zip directory
    :param out_dir: Output directory to write extracted files
    :param rename_extracted_files: Set True to rename extracted files according the original zip file name
    :return: None
    """

    print('Extracting zip files...')
    for zip_file in glob(zip_dir + '*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_extracted_files:
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_file[zip_file.rfind(os.sep) + 1: zip_file.rfind('.')] + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)
                
           
out_dir = ''
zip_dir = outdir
extract_data(zip_dir, out_dir, rename_extracted_files=True)                
