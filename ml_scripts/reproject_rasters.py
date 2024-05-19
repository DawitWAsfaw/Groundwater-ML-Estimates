"""
Author
# Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

from osgeo import gdal
import os
import subprocess
from glob import glob

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
        gdal_path += 'OSGeo4W.bat'
        sys_call = [gdal_path] + [gdal_command] + args
    if verbose:
        print(sys_call)
    return sys_call

def get_raster_extents(gdal_raster):
    """
    Get Raster Extents
    :param gdal_raster: Input gdal raster object
    :return: (Xmin, YMax, Xmax, Ymin)
    """
    transform = gdal_raster.GetGeoTransform()
    ulx, uly = transform[0], transform[3]
    xres, yres = transform[1], transform[5]
    lrx, lry = ulx + xres * gdal_raster.RasterXSize, uly + yres * gdal_raster.RasterYSize
    return str(ulx), str(lry), str(lrx), str(uly)

def reproject_raster(input_raster_file, outfile_path, resampling_factor=1, resampling_func=gdal.GRA_NearestNeighbour,
                     downsampling=True, from_raster=None, keep_original=False, gdal_path='C:/OSGeo4W/', verbose=True):
    """
    Reproject raster using GDAL system call
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param from_raster: Reproject input raster considering another raster
    :param keep_original: Set True to only use the new projection system from 'from_raster'. The original raster extent
    is not changed
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    src_raster_file = gdal.Open(input_raster_file)
    rfile = src_raster_file
    if from_raster and not keep_original:
        rfile = gdal.Open(from_raster)
        resampling_factor = 1
    src_band = rfile.GetRasterBand(1)
    transform = rfile.GetGeoTransform()
    xres, yres = transform[1], transform[5]
    extent = get_raster_extents(rfile)
    dst_proj = rfile.GetProjection()
    no_data = src_band.GetNoDataValue()
    if not downsampling:
        resampling_factor = 1 / resampling_factor
    xres, yres = xres * resampling_factor, yres * resampling_factor

    resampling_dict = {gdal.GRA_NearestNeighbour: 'near', gdal.GRA_Bilinear: 'bilinear', gdal.GRA_Cubic: 'cubic',
                       gdal.GRA_CubicSpline: 'cubicspline', gdal.GRA_Lanczos: 'lanczos', gdal.GRA_Average: 'average',
                       gdal.GRA_Mode: 'mode', gdal.GRA_Max: 'max', gdal.GRA_Min: 'min', gdal.GRA_Med: 'med',
                       gdal.GRA_Q1: 'q1', gdal.GRA_Q3: 'q3'}
    resampling_func = resampling_dict[resampling_func]
    args = ['-t_srs', dst_proj, '-te', extent[0], extent[1], extent[2], extent[3],
            '-dstnodata', str(no_data), '-r', str(resampling_func), '-tr', str(xres), str(yres), '-ot', 'Float32',
            '-overwrite', input_raster_file, outfile_path]
    sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdalwarp', args=args, verbose=verbose)
    subprocess.call(sys_call)




def reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W', verbose=True):
    """
    Reproject rasters in a directory
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param ref_raster: Reference raster file to consider while reprojecting
    :param outdir: Output directory for storing reprojected rasters
    :param pattern: Raster extension
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        reproject_raster(raster_file, from_raster=ref_raster, outfile_path=out_raster, gdal_path=gdal_path,
                         verbose=verbose)
        




ref_raster =  'reproject_ref.tif'
input_raster_dir = 'raster_folder'   
outdir = input_raster_dir + 'reprojected/'

reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='C:/OSGeo4W/', verbose=True)