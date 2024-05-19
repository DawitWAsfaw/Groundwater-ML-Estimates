
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import rasterio as rio
import glob
import os

NO_DATA_VALUE = -32767.0
#======================================================================================================================
def make_gdal_sys_call_str(gdal_path, gdal_command, args, verbose=True):
    """
    Make GDAL system call string
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param gdal_command: GDAL command to use
    :param args: GDAL arguments as a list
    :param verbose: Set True to print system call info
    :return: GDAL system call string,
    """

    sys_call = [gdal_path + gdal_command] + args
    if os.name == 'nt':
        gdal_path += 'C:/OSGeo4W/OSGeo4W.bat'
        sys_call = [gdal_path] + [gdal_command] + args
    if verbose:
        print(sys_call)
    return sys_call

#======================================================================================================================
def reproject_vector(input_vector_file, outfile_path, ref_file, crs='epsg:4326', crs_from_file=True, raster=True):
    """
    Reproject a vector file
    :param input_vector_file: Input vector file path
    :param outfile_path: Output vector file path
    :param crs: Target CRS
    :param ref_file: Reference file (raster or vector) for obtaining target CRS
    :param crs_from_file: If true (default) read CRS from file (raster or vector)
    :param raster: If true (default) read CRS from raster else vector
    :return: Reprojected vector file in GeoPandas format
    """

    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
            ref_file = gpd.read_file(ref_file)
        crs = ref_file.crs
    else:
        crs = {'init': crs}
    output_vector_file = input_vector_file.to_crs(crs)
    output_vector_file.to_file(outfile_path)
    return output_vector_file
#======================================================================================================================
def gdf2shp(input_df, geometry, source_crs, target_crs, outfile_path):
    """
    Convert Geodatafarme to SHP
    :param input_df: Input geodataframe
    :param geometry: Geometry (Point) list
    :param source_crs: CRS of the source file
    :param target_crs: Target CRS
    :param outfile_path: Output file path
    :return:
    """

    crs = {'init': source_crs}
    gdf = gpd.GeoDataFrame(input_df, crs=crs, geometry=geometry)
    gdf.to_file(outfile_path)
    if target_crs != source_crs:
        reproject_vector(outfile_path, outfile_path=outfile_path, crs=target_crs, crs_from_file=False, ref_file=None)
#======================================================================================================================

def csv2shp(input_csv_file, outfile_path, delim=',', source_crs='epsg:4326', target_crs='epsg:4326',
            long_lat_pos=(1, 2)):
    """
    Convert CSV to Shapefile
    :param input_csv_file: Input CSV file path

    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
    :param outfile_path: Output file path
    :param delim: CSV file delimeter
    :param source_crs: CRS of the source file
    :param target_crs: Target CRS
    :param long_lat_pos: Tuple containing positions of longitude and latitude columns respectively (zero indexing)
    :return: None
    """

    input_df = pd.read_csv(input_csv_file, delimiter=delim)
    input_df = input_df.dropna(axis=1)
    long, lat = input_df.columns[long_lat_pos[0]], input_df.columns[long_lat_pos[1]]
    geometry = [Point(xy) for xy in zip(input_df[long], input_df[lat])]
    gdf2shp(input_df, geometry, source_crs, target_crs, outfile_path)
#=====================================================================================================================
def csvs2shps(input_dir, output_dir, pattern='*.csv', target_crs='EPSG:4326', delim=',',
              long_lat_pos=(1, 2)):
    """
    Convert all CSV files present in a folder to corresponding Shapefiles
    :param input_dir: Input directory containing csv files which are named as <Layer_Name>_<Year>.[csv|txt]
    :param output_dir: Output directory
    :param pattern: CSV  file pattern
    :param target_crs: Target CRS
    :param delim: CSV file delimeter
    :param long_lat_pos: Tuple containing positions of longitude and latitude columns respectively (zero indexing)
    :return: None
    """

    for file in glob.glob(input_dir + pattern):
        outfile_path = output_dir + file[file.rfind(os.sep) + 1: file.rfind('.') + 1] + 'shp'
        csv2shp(file, outfile_path=outfile_path, delim=delim, target_crs=target_crs, long_lat_pos=long_lat_pos)
# #======================================================================================================================
