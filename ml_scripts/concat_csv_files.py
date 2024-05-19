
import pandas as pd
import os
from glob import glob


def concat_csv_files(input_csv_file_path, output_file_path, pattern = '*.csv'):
    """
    Concatenates raster extracted yealy csv values for individual variables
    : input_csv_file_path: string path director for input csv files
    :  output_file_path: string path director to store concatenated csv files
    : return: none
    """
    names = []
    for file in glob(input_csv_file_path + pattern):
        variable =os.path.basename(file).split('/')[-1]
        name = variable[variable.rfind(os.sep) + 1: variable.rfind('_')+1]
        names.append(name)
    names = list(set(names))
    
    for name in names:
        print(name)
        df =pd.DataFrame()
        for f in glob(input_csv_file_path + name + pattern):
            df1 = pd.read_csv(f)
            df2 = df1.rename(columns={df1.columns[2]: name})
            df= df.append(df2)
        df.to_csv(output_file_path + name + 'all.csv',index = False)

            
#=================================================================================================        
input_csv_file_path = 'csv_yearly/'
output_file_path = 'csv_concat/'
 
concat_csv_files(input_csv_file_path, output_file_path, pattern = '*.csv')