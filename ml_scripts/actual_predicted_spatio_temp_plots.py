import pandas as pd
from ml_scripts.convercsvToshp import  csvs2shps
from ml_scripts.sumPointvalues_ingrid import aggregate_pointValues_ingrid
from ml_scripts.rasterToDataframe import create_dataframe
from ml_scripts.plotting import spatio_temporal_actual_vs_predicted_rasterplot

actual_pump = pd.read_csv('C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/Final_models/plots/predicted_actual_raster/predicted/predict_pump.csv')
actual_pump.columns
base_dir  = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/Final_models/plots/predicted_actual_raster/'
actual_dir = 'actual/'
csv_folder = 'csv/'
shp_folder = 'shp/'
tiff_folder  = 'tiff/'

for year in actual_pump['year'].unique():
    print(year)
    year_data =actual_pump[actual_pump['year']==year]
    #print(year_data)
    new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','predict_pump']].dropna(axis=0)
    new_data.to_csv(base_dir + actual_dir + csv_folder +  'predict_pump_{}.csv'.format(year),index=False)


shp_grid = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/'\
    'for_monty/for_Monty_updates/shp_files/ref_grid.shp'
ref_rast = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/'\
    'for_monty/for_Monty_updates/ref_rasters/ref_raster_grid.tif'

csvs2shps(input_dir=base_dir + actual_dir + csv_folder, 
          output_dir=base_dir + actual_dir + shp_folder,
          pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))  



aggregate_pointValues_ingrid(input_dir=base_dir + actual_dir + shp_folder,
                               shp_grid = shp_grid,ref_rast=ref_rast, 
                               output_dir= base_dir + actual_dir  + tiff_folder, 
                               pattern = '*.shp') 

test_grid_df = create_dataframe(input_rast_dir = base_dir + actual_dir  + tiff_folder, 
                                  input_grid_file=shp_grid, 
                                  output_dir = base_dir + actual_dir  + tiff_folder, 
                                  column_names=None, pattern='*.tif', 
                                  make_year_col=True, ordering=False, cellid_attr='cellid')

test_grid_df.to_csv(base_dir + 'test_grid_df.csv',index=False)
actual_tiff_file = base_dir + actual_dir  + tiff_folder  + 'actual_pump_2017.tif'



predicted_tiff_file = base_dir + actual_dir  + tiff_folder  + 'predict_pump_2017.tif'

model_input_file = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/'\
    'ml_analysis/python_scripts/Cleaned_up_python_codes/data/ml_input/ml_input_2008_2020.csv'


spatio_temporal_actual_vs_predicted_rasterplot(model_input_file, 
                                                actual_tiff_file, 
                                                predicted_tiff_file, 
                                                test_grid_df, 
                                                base_dir)    
