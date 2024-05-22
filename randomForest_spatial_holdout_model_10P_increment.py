import pandas as pd
import numpy as np
import random
import os
from collections import defaultdict 
import  matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

from ml_scripts.convercsvToshp import  csvs2shps
from ml_scripts.aggregate_pointValues_ingrid import aggregate_pointValues_ingrid
from ml_scripts.rasterToDataframe import create_dataframe

from ml_scripts.plotting import pdp_plots
from ml_scripts.plotting import plot_featureImportance
from ml_scripts.plotting import spatio_temporal_actual_vs_predicted_rasterplot
from ml_scripts.plotting import grid_scale_actual_vs_predicted_timeseries_plot
from ml_scripts.plotting import point_scale_actual_vs_predicted_timeseries_plot
import joblib

import warnings
warnings.filterwarnings("ignore")

no_value = -9999
# print(os.getcwd())




plt.close("all")
def randomForest_spatial_holdout_model(model_input_file, base_dir,shp_grid,ref_rast):
    """
    Runs 17 Random Forest models using testing data ranging from 10 - 90 %  at 10% increment
    ----------
    model_input_file : csv path for data used for building and testing the model
    base_dir : Parent directory path to store csv, shp,tiff data
    shp_grid : 2 km by 2 km grid used as a reference to aggregate withdrawal data
    ref_rast :  Raster data with 2 km spatial resolution used as a reference for converting aggregate data into raster format
    -------
    None.

    """
   
    fract=np.arange(0.10,1.00, 0.10).round(decimals=3)
    np.random.seed(1234)
    num_seeds = fract.size
    seed_values = np.random.choice(range(1000), size=num_seeds + 1, replace=False)
    round_places = 2
    score_metric_dic = defaultdict(list)
    score_metric_dic =  {
      '% Of data Used to train a model' : [],
      '# Of Training Wells': [],
      '# Of Testing Wells':[],
     
      'Train Point R2':[],
      'Train Point MAE(mm)' : [],
      'Train Point RMSE(mm)' : [],
      'Train Point ME(mm)' : [],
     
      'Test Point R2' : [],
      'Test Point MAE(mm)' : [],
      'Test Point RMSE(mm)' : [],
      'Test Point ME(mm)' : [],
     
      
      'Train Grid R2': [],
      'Train Grid MAE(mm)' : [],
      'Train Grid RMSE(mm)': [],
      'Train Grid ME(mm)' : [],
     
      'Test Grid R2': [],
      'Test Grid MAE(mm)' : [],
      'Test Grid RMSE(mm)':  [],
      'Test Grid ME(mm)' : [],
     
     
      'Train Grid Mean R2' :  [],
      'Train Grid Mean MAE(mm)': [],
      'Train Grid Mean RMSE(mm)' : [],
      'Train Grid Mean ME(mm)' : [],
     
      'Test Grid Mean R2' :  [],
      'Test Grid Mean MAE(mm)': [],
      'Test Grid Mean RMSE(mm)' : [],
      'Test Grid Mean ME(mm)' : []
     
      } 
    

    model_input_data = pd.read_csv(model_input_file)
    model_input_data.dtypes.value_counts()
    
    model_input_data.replace([np.inf,-np.inf],np.nan,inplace = True)
    model_input_data.dropna(axis =0,inplace = True)
    
    data_split_name = pd.DataFrame(model_input_data['gridid'].unique(),columns=['gridid'])
    
    train_yearly_aver = pd.DataFrame()
    test_yearly_aver = pd.DataFrame()
    actual_pred_pt = pd.DataFrame()
    train_id_data,test_id = train_test_split(data_split_name,test_size=0.1,random_state=111)
    
    fract = np.array([0] + fract.tolist())
    
    for num, seed in zip(fract, seed_values):
        if num == 0:
            train_id = train_id_data
        else:
            train_id,unused_id = train_test_split(train_id_data,test_size=num,random_state=seed)
        # print(num)
        score_metric_dic['% Of data Used to train a model'].append(100 -(num*100))
        num_str = str(int((num*100).round(0)))
        # print(num_str)
        folder_names = num_str + '_percent/'
        train_fold= 'train/'
        test_fold =  'test/'
        actual_fold = 'actual/'
        predicted_fold = 'predicted/'
        csv_folder = 'csv/'
        model_folder  = 'model/'
        shp_folder = 'shp/'
        tiff_folder = 'tiff/'
        plot_folder = 'plots/'
        train_dir = folder_names + train_fold
        test_dir = folder_names + test_fold
        actual_dir  = folder_names + actual_fold
        predicted_dir = folder_names + predicted_fold 
        model_dir = base_dir + folder_names + model_folder
        plots_dir = base_dir + folder_names + plot_folder
        
#  ================================================================================================================  
#   Split training and testing data based on unique grid id     
#  ================================================================================================================  
        # print(train_id.shape)
        # print(test_id.shape)
    
        train_data = pd.DataFrame()
        for id in train_id['gridid'].unique():
            selected_train_data = model_input_data[model_input_data['gridid'] == id]
            train_data = pd.concat([train_data, selected_train_data])
                
            
        num_train_wells =  train_data.value_counts('PDIV_ID').count()
        score_metric_dic['# Of Training Wells'].append(num_train_wells)
        
        test_data = pd.DataFrame()
       
        for id in test_id['gridid']:
            selected_test_data = model_input_data[model_input_data['gridid'] == id]
            test_data = pd.concat([test_data, selected_test_data])
            
        num_test_wells =  test_data.value_counts('PDIV_ID').count()
        score_metric_dic['# Of Testing Wells'].append(num_test_wells) 
        
#  ================================================================================================================  
#   Removing column names not included in training and creating training set    
#  ================================================================================================================  
     
        
        drop_columns_x = ['PDIV_ID','gridid', 'year', 'long_nad83', 'lat_nad83', 'pump_aft_acre','irr_area_mm2','pump_mm','pump_mm3']
        
        train_x = train_data.drop(drop_columns_x,axis=1)
        train_x.to_csv(base_dir + folder_names + 'train_x_{}P.csv'.format(num_str ),index=False)
        
        train_y = train_data['pump_mm']
        train_y.to_csv(base_dir + folder_names + 'train_y_{}P.csv'.format(num_str ),index=False)
        
        train_y_mean =pd.DataFrame(train_data['pump_mm'],columns=['pump_mm'])
        train_y_mean_value = train_data['pump_mm'].mean()
        train_y_mean['meanTr_mm'] = train_y_mean_value 
        train_y_mean = train_y_mean['meanTr_mm']
        train_y_mean.to_csv(base_dir + folder_names + 'train_y_mean_{}P.csv'.format(num_str ),index=False)
       
#  ================================================================================================================  
#   Creating testing sets
#  ================================================================================================================  
           
        test_x = test_data.drop(drop_columns_x,axis=1)
        
        test_y = test_data['pump_mm'] 
        test_y.to_csv(base_dir + folder_names + 'test_y_{}P.csv'.format(num_str ),index=False)
       
        
        test_y_mean =pd.DataFrame(test_data['pump_mm'],columns=['pump_mm'])
        test_y_mean_value = test_data['pump_mm'].mean()
        test_y_mean['meanTe_mm'] = test_y_mean_value  
        test_y_mean = test_y_mean['meanTe_mm']
        test_y_mean.to_csv(base_dir + folder_names + 'test_y_mean_{}P.csv'.format(num_str),index=False)
       
#  ================================================================================================================  
#    Runinning parameter tuning using grid search - optional - 
#  ================================================================================================================  
                     
        # model_param = {'max_features': [4, 5, 6, 7]}
        
        # rf_model =RandomForestRegressor(random_state=0, n_jobs=-1)
        
        # ranFor_model= GridSearchCV(estimator=rf_model, param_grid= model_param, cv = 5,verbose = 2, n_jobs=-1)
        # ranFor_model.fit(train_x,train_y)
        # print(ranFor_model.best_params_)
        # ranFor_model.best_params_
        
#  ================================================================================================================  
#  Runinning random forest model using best parameter values
#  ================================================================================================================        
      
    
        ranFor_model_optimized =RandomForestRegressor(max_features= 4, 
                                                      n_estimators= 200,
                                                      random_state=111, 
                                                      n_jobs=-1)
        
        train_x.describe().to_csv(f'{model_dir}train_x_{num}P_{seed}.csv')
        train_y.describe().to_csv(f'{model_dir}train_y_{num}P_{seed}.csv')


        ranFor_model_optimized.fit(train_x,train_y)
        with open(model_dir + 'ranFor_model_{}P.joblib'.format(num_str),'wb') as f:
            joblib.dump(ranFor_model_optimized ,f)
            
#  ================================================================================================================  
# Create ml anlaysis plots - PD and FI
        print('Plotting partial dependence .....')
        
        pdp_plots(ranFor_model_optimized,
                  train_x,
                  train_y,
                  plots_dir)
        plt.close('All')
        print('Plotting feature importance .....')
        plot_featureImportance(ranFor_model_optimized, 
                                train_x, 
                                plots_dir)
        plt.close('All')
#  ================================================================================================================  
        print('Calculating Point Scale model Train Error metrics')    
        train_predict_mm = ranFor_model_optimized.predict(train_x)
        train_predict_pt = pd.Series(train_predict_mm, name='train_predict_mm') 
        train_actual_y = train_y
        train_predict_y= train_predict_pt
      
        
        trai_pt_score = ranFor_model_optimized.score(train_x, train_y)
        print(trai_pt_score)
        trai_pt_mae = metrics.mean_absolute_error(train_actual_y, train_predict_y)
        trai_pt_rmse = metrics.mean_squared_error(train_actual_y, train_predict_y, squared=False)
        trai_pt_me = np.mean(train_actual_y - train_predict_y)
        
        trai_pt_rmse = np.round(trai_pt_rmse, round_places)
        trai_pt_mae = np.round(trai_pt_mae, round_places)
        trai_pt_me = np.round(trai_pt_me, round_places)
        
        score_metric_dic['Train Point R2'].append(trai_pt_score)
        score_metric_dic['Train Point MAE(mm)'].append(trai_pt_mae)
        score_metric_dic['Train Point RMSE(mm)'].append(trai_pt_rmse)
        score_metric_dic['Train Point ME(mm)'].append(trai_pt_me)
# ================================================================================================================
        print('Calculating Point Scale model Test Error metrics')    
        test_predict_mm = ranFor_model_optimized.predict(test_x)
        tes_pt_score = ranFor_model_optimized.score(test_x, test_y)
        test_predict_pt = pd.Series(test_predict_mm, name='test_predict_mm')
        # print(num_str)
        # print(test_predict_pt.describe())
        test_actual_y = test_y
        test_predict_y= test_predict_pt
       
        
        tes_pt_mae = metrics.mean_absolute_error(test_actual_y, test_predict_y)
        tes_pt_rmse = metrics.mean_squared_error(test_actual_y, test_predict_y, squared=False)
        tes_pt_me = np.mean(test_actual_y -test_predict_y)
        
        tes_pt_score = np.round(tes_pt_score, round_places)
        tes_pt_rmse = np.round(tes_pt_rmse, round_places)
        tes_pt_mae = np.round(tes_pt_mae, round_places)
        tes_pt_me = np.round(tes_pt_me, round_places)
        
        score_metric_dic['Test Point R2'].append(tes_pt_score)
        score_metric_dic['Test Point MAE(mm)'].append(tes_pt_mae)
        score_metric_dic['Test Point RMSE(mm)'].append(tes_pt_rmse)
        score_metric_dic['Test Point ME(mm)'].append(tes_pt_me)


#  ================================================================================================================  
#      Actual training data used for development of randforest model
        train_actual = train_data[[ 'gridid','year', 'long_nad83', 'lat_nad83','pump_mm3']]
        train_actual = train_actual.rename(columns={'pump_mm3':"actualTr_mm3" }, errors="raise")

        train_actual.to_csv(base_dir + train_dir + 'train_actual_{}P.csv'.format(num_str ),index=False)
        # train_actual - disaggregate to individual years
   
        for year in train_actual['year'].unique():
            year_data = train_actual[train_actual['year']==year]
            new_data = year_data[['gridid', 'long_nad83', 'lat_nad83','year','actualTr_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + train_dir + csv_folder + 'train_actual_{}P_{}.csv'.format(num_str , year),index=False)
           
        csvs2shps(input_dir=base_dir + train_dir + csv_folder, 
                  output_dir=base_dir + train_dir + shp_folder,
                  pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2)) 
        
#  ================================================================================================================  
#       Predicted training data 
        train_predict_mm = ranFor_model_optimized.predict(train_x)
        train_predict= train_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
        train_predict['train_predict_mm']=  train_predict_mm
        
        train_predict['predTr_mm3'] =train_predict['train_predict_mm']* train_predict['irr_area_mm2']
        train_predict.to_csv(base_dir + train_dir + 'train_predict_{}P.csv'.format(num_str ),index=False)
            
            # train_predict - disaggregate to individual years
    
        for year in train_predict['year'].unique():
            year_data =train_predict[train_predict['year']==year]
            #print(year_data)
            new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','predTr_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + train_dir + csv_folder + 'train_predict_{}P_{}.csv'.format(num_str, year),index=False)
            
            
            
#  ================================================================================================================  
        train_mean_pump = train_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
        train_mean_pump['pump_mean'] = train_y_mean
        train_mean_pump['meanTr_mm3'] =train_mean_pump['pump_mean']* train_mean_pump['irr_area_mm2']
        train_mean_pump.to_csv(base_dir + train_dir + 'train_mean_{}P.csv'.format(num_str ),index=False)
        
        for year in train_mean_pump['year'].unique():
            year_data =train_mean_pump[train_mean_pump['year']==year]
            #print(year_data)
            new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','meanTr_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + train_dir + csv_folder + 'train_mean_{}P_{}.csv'.format(num_str , year),index=False)
            
        print('Converting csv to shp....')
        csvs2shps(input_dir=base_dir + train_dir + csv_folder, 
                  output_dir=base_dir + train_dir + shp_folder,
                  pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))  
       
        print('aggregating values in a grid....')
      
        aggregate_pointValues_ingrid(input_dir=base_dir + train_dir + shp_folder,
                                      shp_grid = shp_grid,ref_rast=ref_rast, 
                                      output_dir= base_dir + train_dir  + tiff_folder, 
                                      pattern = '*.shp') 
        print('converting gridded rater value to df....')
        
        train_grid_df= create_dataframe(input_rast_dir = base_dir + train_dir  + tiff_folder, 
                                          input_grid_file=shp_grid, 
                                          output_dir = base_dir + train_dir  + tiff_folder, 
                                          column_names=None, pattern='*.tif', 
                                          make_year_col=True, ordering=False, cellid_attr='cellid')

       
        train_grid_df.columns.values[3]  =  'wateryear'
        
        train_grid_ave = train_grid_df.groupby('wateryear').mean().reset_index()
        train_yearly_aver = pd.concat([train_yearly_aver,train_grid_ave],axis=1)
#  ================================================================================================================  

        print('Calculating Grid Scale model Train Error metrics')
        
        train_grid_df.to_csv(base_dir + train_dir  + tiff_folder + 'training_{}.csv'.format(num_str ),index=False)
        train_grid_actual_y  = train_grid_df[train_grid_df.columns[0]]
        train_grid_actual_y = train_grid_actual_y
        
        train_grid_mean_y  = train_grid_df[train_grid_df.columns[1]]
        train_grid_mean_y = train_grid_mean_y
        
        train_grid_predict_y  = train_grid_df[train_grid_df.columns[2]]
        train_grid_predict_y = train_grid_predict_y
        
        trai_grid_score = metrics.r2_score(train_grid_df[train_grid_df.columns[0]],train_grid_df[train_grid_df.columns[2]])
        trai_grid_mae = metrics.mean_absolute_error(train_grid_actual_y, train_grid_predict_y)
        trai_grid_rmse = metrics.mean_squared_error(train_grid_actual_y, train_grid_predict_y , squared=False)
        trai_grid_me = np.mean(train_grid_actual_y - train_grid_predict_y)
        
        trai_grid_rmse = np.round(trai_grid_rmse, round_places)
        trai_grid_mae = np.round(trai_grid_mae, round_places)
        trai_grid_me = np.round(trai_grid_me, round_places)
        
        score_metric_dic['Train Grid R2'].append(trai_grid_score)
        score_metric_dic['Train Grid MAE(mm)'].append(trai_grid_mae)
        score_metric_dic['Train Grid RMSE(mm)'].append(trai_grid_rmse)
        score_metric_dic['Train Grid ME(mm)'].append(trai_grid_me)
        
#  ================================================================================================================    
        print('Calculating Grid Scale model Error metrics using withdrawal mean as predicted value')
        trai_mean_grid_score = metrics.r2_score(train_grid_df[train_grid_df.columns[0]],train_grid_df[train_grid_df.columns[1]])   
        trai_grid_mean_mae = metrics.mean_absolute_error(train_grid_actual_y,train_grid_mean_y)
        trai_grid_mean_rmse = metrics.mean_squared_error(train_grid_actual_y,train_grid_mean_y, squared=False)
        trai_grid_mean_me = np.mean(train_grid_actual_y-train_grid_mean_y)
        
        trai_grid_mean_rmse = np.round(trai_grid_mean_rmse, round_places)
        trai_grid_mean_mae = np.round(trai_grid_mean_mae, round_places)
        trai_grid_mean_me = np.round(trai_grid_mean_me, round_places)
        
        score_metric_dic['Train Grid Mean R2'].append(trai_mean_grid_score)
        score_metric_dic['Train Grid Mean MAE(mm)'].append(trai_grid_mean_mae)
        score_metric_dic['Train Grid Mean RMSE(mm)'].append(trai_grid_mean_rmse)
        score_metric_dic['Train Grid Mean ME(mm)'].append(trai_grid_mean_me)
    
#  ================================================================================================================  
#     Actual Testing data to test randforest model
        test_actual = test_data[['gridid', 'long_nad83', 'lat_nad83','year','pump_mm3']]
        test_actual = test_actual.rename(columns={'pump_mm3': "actualTe_mm3"}, errors="raise")

        test_actual.to_csv(base_dir + test_dir + 'test_actual_{}P.csv'.format(num_str ),index=False)
        
        # Test_Actual- disaggregate to individual years

        for year in test_actual['year'].unique():
            year_data =test_actual[test_actual['year']==year]
            #print(year_data)
            new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','actualTe_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + test_dir + csv_folder +  'test_actual_{}P_{}.csv'.format(num_str , year),index=False)
        
        
        
#  ================================================================================================================  
#      Predicted Test data to assess model performance for unfamiliar dataset
        test_predict_mm = ranFor_model_optimized.predict(test_x)

        test_predict = test_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
        test_predict['predTe_mm']=  test_predict_mm
#  ================================================================================================================
# actual and predicted point scale model - csv in put for time series plots
        
        test_actual_pt = test_data[['PDIV_ID','gridid', 'long_nad83', 'lat_nad83','year']]
        test_act_mm = test_data['pump_mm']
        
        test_actual_pt['pump_mm _{}'.format(num_str)] = test_act_mm
        
        test_actual_pt['predTe_mm_{}'.format(num_str)] =  test_predict_mm
           
        actual_pred_pt=  pd.concat([actual_pred_pt,test_actual_pt],axis=1)
#  ================================================================================================================
        
        test_predict['predTe_mm3'] = test_predict['irr_area_mm2']* test_predict['predTe_mm']
        
        test_predict =test_predict.drop(['irr_area_mm2','predTe_mm'],axis=1)
        
        test_predict.to_csv(base_dir + test_dir + 'test_predict_{}P.csv'.format(num_str ),index=False)
        
        
        for year in test_predict['year'].unique():
            year_data = test_predict[test_predict['year']==year]
            new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','predTe_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + test_dir + csv_folder + 'test_predict_{}P_{}.csv'.format(num_str ,year),index=False)
            
        
#  ================================================================================================================  
        test_mean_pump = test_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
        test_mean_pump['pump_mean'] = test_y_mean
        test_mean_pump['meanTe_mm3'] =test_mean_pump['pump_mean']* test_mean_pump['irr_area_mm2']
        test_mean_pump.to_csv(base_dir + test_dir + 'test_mean_{}P.csv'.format(num_str ),index=False)
        
        for year in test_mean_pump['year'].unique():
            year_data =test_mean_pump[test_mean_pump['year']==year]
            #print(year_data)
            new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','meanTe_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + test_dir + csv_folder + 'test_mean_{}P_{}.csv'.format(num_str , year),index=False)
            
        
        csvs2shps(input_dir=base_dir + test_dir + csv_folder, 
                  output_dir=base_dir + test_dir + shp_folder,
                  pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))
        
        
        aggregate_pointValues_ingrid(input_dir=base_dir + test_dir + shp_folder,
                                      shp_grid = shp_grid,ref_rast=ref_rast, 
                                      output_dir= base_dir + test_dir + tiff_folder, 
                                      pattern = '*.shp') 
        
        test_grid_df = create_dataframe(input_rast_dir = base_dir + test_dir  + tiff_folder, 
                                          input_grid_file=shp_grid, 
                                          output_dir = base_dir + test_dir  + tiff_folder, 
                                          column_names=None, pattern='*.tif', 
                                          make_year_col=True, ordering=False, cellid_attr='cellid')
    

#  ================================================================================================================  
# Calculating annual average withdrawals 
        test_grid_df.columns.values[3]  =  'wateryear' 
        test_grid_ave = test_grid_df.groupby('wateryear').mean().reset_index()
        test_yearly_aver = pd.concat([test_yearly_aver,test_grid_ave],axis=1)
        
        test_grid_df.to_csv(base_dir + test_dir  + tiff_folder + 'testing_{}.csv'.format(num_str ),index=False)
       
    
        
#  ================================================================================================================  
# Calculating grid scale model test error metrics
        print('Calculating Grid Scale model Test Error metrics')

        test_grid_actual_y  = test_grid_df[test_grid_df.columns[0]]
        test_grid_actual_y = test_grid_actual_y
        
        test_grid_mean_y  = test_grid_df[test_grid_df.columns[1]]
        test_grid_mean_y = test_grid_mean_y
        
        test_grid_predict_y  = test_grid_df[test_grid_df.columns[2]]
        test_grid_predict_y = test_grid_predict_y
        
        tes_grid_score = metrics.r2_score(test_grid_df[test_grid_df.columns[0]],test_grid_df[test_grid_df.columns[2]])
        tes_grid_mae = metrics.mean_absolute_error(test_grid_actual_y, test_grid_predict_y)
        tes_grid_rmse = metrics.mean_squared_error(test_grid_actual_y, test_grid_predict_y, squared=False)
        tes_grid_me = np.mean(test_grid_actual_y- test_grid_predict_y)
        
        tes_grid_rmse = np.round(tes_grid_rmse, round_places)
        trai_grid_mae = np.round(tes_grid_mae, round_places)
        trai_grid_me = np.round(tes_grid_me, round_places)
        
        score_metric_dic['Test Grid R2'].append(tes_grid_score)
        score_metric_dic['Test Grid MAE(mm)'].append(tes_grid_mae)
        score_metric_dic['Test Grid RMSE(mm)'].append(tes_grid_rmse)
        score_metric_dic['Test Grid ME(mm)'].append(tes_grid_me)
#  ================================================================================================================  
        print('Calculating Grid Scale model Error metrics using withdrawal mean as predicted value')
        tes_grid_mean_score =  metrics.r2_score(test_grid_df[test_grid_df.columns[0]],test_grid_df[test_grid_df.columns[1]])
        tes_grid_mean_mae = metrics.mean_absolute_error(test_grid_actual_y,test_grid_mean_y)
        tes_grid_mean_rmse = metrics.mean_squared_error(test_grid_actual_y,test_grid_mean_y, squared=False)
        tes_grid_mean_me = np.mean(test_grid_actual_y-test_grid_mean_y)
        
        tes_grid_mean_rmse = np.round(tes_grid_mean_rmse, round_places)
        tes_grid_mean_mae = np.round(tes_grid_mean_mae, round_places)
        tes_grid_mean_me = np.round(tes_grid_mean_me, round_places)
        
        score_metric_dic['Test Grid Mean R2'].append(tes_grid_mean_score)
        score_metric_dic['Test Grid Mean MAE(mm)'].append(tes_grid_mean_mae)
        score_metric_dic['Test Grid Mean RMSE(mm)'].append(tes_grid_mean_rmse)
        score_metric_dic['Test Grid Mean ME(mm)'].append(tes_grid_mean_me)
#  ================================================================================================================  
  # Creating a raster file of the actual data 
        train_actual = train_actual.rename(columns={'actualTr_mm3':'actual_mm3' }, errors="raise")
        test_actual = test_actual.rename(columns={'actualTe_mm3': 'actual_mm3'}, errors="raise")

        actual = pd.concat([train_actual, test_actual], axis=0)
        actual.to_csv(base_dir + train_dir + 'train_actual_{}P.csv'.format(num_str ),index=False)
        # train_actual - disaggregate to individual years
    
        for year in actual['year'].unique():
            year_data = actual[actual['year']==year]
            new_data = year_data[['gridid', 'long_nad83', 'lat_nad83','year','actual_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + actual_dir + csv_folder + 'actual_{}P_{}.csv'.format(num_str , year),index=False)
           
        csvs2shps(input_dir=base_dir + actual_dir + csv_folder, 
                  output_dir=base_dir + actual_dir + shp_folder,
                  pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))
        
        aggregate_pointValues_ingrid(input_dir=base_dir + actual_dir + shp_folder,
                                      shp_grid = shp_grid,ref_rast=ref_rast, 
                                      output_dir= base_dir + actual_dir + tiff_folder, 
                                      pattern = '*.shp') 
#  ================================================================================================================  
  # Creating a raster file of the predicted data
        train_predict = train_predict.rename(columns={'predTr_mm3':'predicted_mm3'}, errors="raise")
        test_predict = test_predict.rename(columns={'predTe_mm3':'predicted_mm3'}, errors="raise")
         
        predicted = pd.concat([ train_predict, test_predict], axis=0)
        
        for year in predicted['year'].unique():
            year_data = predicted[predicted['year']==year]
            new_data = year_data[['gridid', 'long_nad83', 'lat_nad83','year','predicted_mm3']].dropna(axis=0)
            new_data.to_csv(base_dir + predicted_dir + csv_folder + 'predicted_{}P_{}.csv'.format(num_str , year),index=False)
        print('converting csv files to shp files')
        csvs2shps(input_dir=base_dir + predicted_dir + csv_folder, 
                  output_dir=base_dir + predicted_dir + shp_folder,
                  pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))
        print('aggregating withdrawal values in a 2 km  grid cell and converting shp files to raster')
        aggregate_pointValues_ingrid(input_dir=base_dir + predicted_dir + shp_folder,
                                      shp_grid = shp_grid,ref_rast=ref_rast, 
                                      output_dir= base_dir + predicted_dir + tiff_folder, 
                                      pattern = '*.shp') 
        
#  ================================================================================================================  
        print('Plotting actual vs predicted raster plots')
        
        
        actual_tiff_file = base_dir + actual_dir  + tiff_folder  + 'actual_{}P_2017.tif'.format(num_str)
   
      
      
        predicted_tiff_file = base_dir + predicted_dir  + tiff_folder  + 'predicted_{}P_2017.tif'.format(num_str)
        
        
        spatio_temporal_actual_vs_predicted_rasterplot(model_input_file, 
                                                        actual_tiff_file, 
                                                        predicted_tiff_file, 
                                                        test_grid_df, 
                                                        plots_dir)    
        

#  ================================================================================================================  

    score_metric_df = pd.DataFrame(score_metric_dic).round(round_places) 
    score_metric_df.to_csv(base_dir + 'Model_error_metrices_5_13_24.csv',index=False)
       
    
    print(score_metric_df)

#  ================================================================================================================  
   
    train_yearly_aver.columns.values[0] = 'wateryear' 
    test_yearly_aver.columns.values[0] = 'wateryear' 
    
    test_yearly_aver = test_yearly_aver.rename(columns={'test_actual_0P':'test_actual','test_mean_0P':'test_mean'}, errors="raise")
    
    
    train_yearly_column = ['wateryear',
                            'train_actual_0P','train_predict_0P','train_mean_0P',
                            'train_actual_10P','train_predict_10P','train_mean_10P',
                            'train_actual_20P','train_predict_20P', 'train_mean_20P',
                            'train_actual_30P','train_predict_30P', 'train_mean_30P',
                            'train_actual_40P','train_predict_40P', 'train_mean_40P',		
                            'train_actual_50P','train_predict_50P', 'train_mean_50P',	
                            'train_actual_60P','train_predict_60P', 'train_mean_60P',
                            'train_actual_70P','train_predict_70P', 'train_mean_70P',
                            'train_actual_80P','train_predict_80P', 'train_mean_80P',	
                            'train_actual_90P','train_predict_90P','train_mean_90P']
    
    
    test_yearly_column = ['wateryear',
                            'test_actual', 'test_mean',
                            'test_predict_0P', 
                            'test_predict_10P',  
                            'test_predict_20P',
                            'test_predict_30P', 
                            'test_predict_40P', 	
                            'test_predict_50P',  
                            'test_predict_60P',
                            'test_predict_70P',
                            'test_predict_80P',	
                            'test_predict_90P']
    
    train_yearly_aver = train_yearly_aver[train_yearly_column]
    test_yearly_aver = test_yearly_aver[test_yearly_column]
    
    train_yearly_aver.to_csv(base_dir + 'train_yearly_aver.csv',index=False)
    test_yearly_aver.to_csv(base_dir + 'test_yearly_aver.csv',index=False)

    test_yearly_aver = pd.read_csv(base_dir + 'test_yearly_aver.csv')
    train_yearly_aver = pd.read_csv(base_dir + 'train_yearly_aver.csv')
    
    
    train_yearly_aver.drop(train_yearly_aver.columns[1:10], inplace=True, axis=1)  
    
    test_yearly_aver.drop(test_yearly_aver.columns[1:10], inplace=True, axis=1)
    
    train_yearly_aver.to_csv(base_dir + 'train_yearly_aver_5_13_24.csv',index=False)
    test_yearly_aver.to_csv(base_dir + 'test_yearly_aver_5_13_24.csv',index=False)
#  ================================================================================================================      
    actual_pred_pt.columns.values[4] = 'wateryear' 
    actual_pred_pt.columns.values[0] ='WELL_ID'
    actual_pred_pt_column = ['WELL_ID',
                             'wateryear',
                             'pump_mm _0',
                             'predTe_mm_10', 
                             'predTe_mm_20',  
                             'predTe_mm_30',
                             'predTe_mm_40', 
                             'predTe_mm_50', 	
                             'predTe_mm_60',  
                             'predTe_mm_70',
                             'predTe_mm_80',
                             'predTe_mm_90']
    actual_pred_pt =  actual_pred_pt[actual_pred_pt_column]
    
    actual_pred_pt.to_csv(base_dir + 'actual_pred_pt.csv',index=False)
#  ================================================================================================================  

    
    print('Plotting actual vs predicted  time series')
   
    grid_scale_actual_vs_predicted_timeseries_plot(test_yearly_aver, base_dir)
    point_scale_actual_vs_predicted_timeseries_plot(actual_pred_pt, base_dir)
    
    fig, ax = plt.subplots()
    ax.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_90P'], linewidth=1,color ='red',label='Predicted')

    ax.set(xlabel=' Year', ylabel='Groundwater withdrawals [mm]')
    ax.annotate('a)',xy=(2008, 47),fontsize="20")
    ax.legend(loc='upper right',frameon =False)
    ax.set_facecolor("white")

    fig.savefig(base_dir + 'test_predict_90P.png')

   

#  ================================================================================================================  
     
shp_grid = '/shp_files/ref_grid.shp'
ref_rast = '/ref_rasters/ref_raster_grid.tif'
    
model_input_file = ' '

base_dir = ' '


randomForest_spatial_holdout_model(model_input_file,base_dir,shp_grid,ref_rast)


