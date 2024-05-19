import os
from collections import defaultdict
from glob import glob
import numpy as np
import pandas as pd
import rasterio as rio

NO_DATA_VALUE = -9999

#=======================================================================================================================
def reindex_df(df, column_names, ordering=False):
    """
    Reindex dataframe columns
    :param df: Input dataframe
    :param column_names: Dataframe column names, these must be df headers
    :param ordering: Set True to apply ordering
    :return: Reindexed dataframe
    """
    if not column_names:
        column_names = df.columns
        ordering = True
    if ordering:
        column_names = sorted(column_names)
    return df.reindex(column_names, axis=1)
#==================================================================================================================

#==================================================================================================================                   
def read_raster_as_arr(raster_file, band=1, get_file=True, rasterio_obj=False, change_dtype=True):
    """
    Get raster array
    :param raster_file: Input raster file path
    :param band: Selected band to read (Default 1)
    :param get_file: Get rasterio object file if set to True
    :param rasterio_obj: Set true if raster_file is a rasterio object
    :param change_dtype: Change raster data type to float if true
    :return: Raster numpy array and rasterio object file (get_file=True and rasterio_obj=False)
    """

    if not rasterio_obj:
        raster_file = rio.open(raster_file)
    else:
        get_file = False
    raster_arr = raster_file.read(band)
    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan
    if get_file:
        return raster_arr, raster_file
    return raster_arr
#==================================================================================================================                
def create_dataframe(input_rast_dir, input_grid_file, output_dir, column_names=None, pattern='*.tif', 
                     make_year_col=True, ordering=False, cellid_attr='cellid'):
    """
    Create dataframe from file list
    :param input_file_dir: Input directory where the file names begin with <Variable>_<Year>, e.g, ET_2015.tif
    :param input_gmd_file: Input GMD shape file
    :param output_dir: Output directory
    :param column_names: Dataframe column names, these must be df headers
    :param pattern: File pattern to look for in the folder
    :param make_year_col: Make a dataframe column entry for year
    :param ordering: Set True to order dataframe column names
    :param cellid_attr: cellid attribute present in the shapefile
    :return: GMD Numpy array
    :return: Pandas dataframe
    """

    raster_file_dict = defaultdict(lambda: [])
    for f in glob(input_rast_dir + pattern):
        sep = f.rfind('_')
        variable, year = f[f.rfind(os.sep) + 1: sep], f[sep + 1: f.rfind('.')]
        print(variable)
        print(year)
        raster_file_dict[int(float(year))].append(f)
    raster_dict = {}
    flag = False
    years = sorted(list(raster_file_dict.keys()))
    df = None
    raster_arr = None
    for year in years:
        file_list = raster_file_dict[year]
        for raster_file in file_list:
            print(raster_file)
            raster_arr = read_raster_as_arr(raster_file, get_file=False)
            raster_arr = raster_arr.reshape(raster_arr.shape[0] * raster_arr.shape[1])
            variable = raster_file[raster_file.rfind(os.sep) + 1: raster_file.rfind('_')]
            raster_dict[variable] = raster_arr
            print(raster_arr.shape)
        if make_year_col:
            raster_dict['year'] = [year] * raster_arr.shape[0]
        if not flag:
            df = pd.DataFrame(data=raster_dict)
            flag = True
        else:
            df = df.append(pd.DataFrame(data=raster_dict))

    df = df.dropna(axis=0)
    df = reindex_df(df, column_names=column_names, ordering=ordering)
    # out_df = output_dir + 'grid_dataframe.csv'
    # df.to_csv(out_df, index=False)
    return df
