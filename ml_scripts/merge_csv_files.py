import pandas as pd
import numpy as np
from glob import glob
import os

def merge_csvUsing_wellId(csv_dir, output_dir,pattern = '*.csv'):
    """
    Creates a single csv file to be ingested into machine learning algorithm
    : csv_dir: string path director for input csv files
    :  output_dir: string path director to store the final csv file
    : return: none
    """
    old_names = []
    new_names = []
    for file in glob(csv_dir  +  pattern):
        df = pd.read_csv(file)
        # print(df.columns)
        variable =os.path.basename(file).split('/')[-1]
        name = variable[variable.rfind(os.sep) + 1: variable.rfind('_')]
        name1 = variable[variable.rfind(os.sep) + 1: variable.rfind('_') + 1]
        new_names.append(name)
        old_names.append(name1)
        
    old_names = list(set(old_names))
    new_names = list(set(new_names))
    old_names= list(map(lambda x: x.replace('pump_', 'pump_mm'),old_names))
    print(old_names)
    print(new_names)
    

    col_renames =  ['cellid','PDIV_ID','year',  'long_nad83', 'lat_nad83', 
                    'aquiThick', 'cropwaterdem', 'evap', 'gwlchan',
                    'gwl', 'ppt', 'pump_aft_acre', 'pump_mm','irr_area_mm2', 
                    'pump_mm3', 'index_right',  'slope', 'tmax','tmin']
           
    
    ml_input = pd.DataFrame()
    
    ml_input = pd.DataFrame()
    for file in glob(csv_dir + pattern):
        df = pd.read_csv(file)
        if  ml_input.empty:
            ml_input = df
        else:
            ml_input = pd.merge(ml_input, df, how='inner', on=['PDIV_ID','year'])
  
    renaming_dic = {'ppt_':'ppt',
                    'tmin_':'tmin',
                    'tmax_':'tmax',
                    'evap_':'evap',
                    'slope_':'slope',
                    'gwl_':'gwl',
                    'gwlchange_':'gwlchan',
                    'aquifThick_':'aquiThick', 
                    'cropwaterDem':'cropwaterdem'
                    
                    }
  
    ml_input.rename(columns=renaming_dic, inplace=True)
    print(ml_input.columns)
    ml_input = ml_input.reindex(columns=col_renames)
    ml_input.replace([np.inf, -np.inf], np.nan, inplace=True)
    ml_input=  ml_input.drop(['index_right'],axis=1)
    ml_input.dropna(inplace=True)
    ml_input.to_csv(output_dir + 'ml_input.csv', index = False)
    

