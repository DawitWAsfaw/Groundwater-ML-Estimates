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
from ml_scripts.plotting import temporal_actual_vs_predicted_rasterplot
import joblib
import warnings

warnings.filterwarnings("ignore")
no_value = -9999
print(os.getcwd())
random.seed(111)

plt.close("all")

   
def randomForest_temporal_holdout_model(model_input_file, temporal_base_dir,test_years,shp_grid,ref_rast):
    """
     Build temporal prediction random forest model
    :model_input_file:  csv file path name with all the variables
    :temporal_base_dir: directory path to store all the ml anlaysis
    : test_years: years used for testing the data #  test_year format = (2008,) for a single year
    : shp_grid: 2 km by 2km grid shape file used to aggregate actual and predicted values
    : ref_rast: a reference raster used to create a 2 km by 2km grid polygon
    :return: none
    """

    round_places = 2
    score_metric_dic = defaultdict(list)
    score_metric_dic =  {
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
#================================================================================================================  
    model_input_data = pd.read_csv(model_input_file)
    #print(model_input_data.columns)
    model_input_data.replace([np.inf,-np.inf],np.nan,inplace = True)
    model_input_data.dropna(axis =0,inplace = True)
   
#================================================================================================================   
    train_fold= 'train/'
    test_fold =  'test/'
    actual_fold = 'actual/'
    predicted_fold = 'predicted/'
    csv_folder = 'csv/'
    model_folder  = 'model/'
    shp_folder = 'shp/'
    tiff_folder = 'tiff/'
    plot_folder = 'plots/'
    temporal_train_dir = temporal_base_dir + train_fold
    temporal_test_dir = temporal_base_dir+ test_fold
    temporal_model_dir = temporal_base_dir  + model_folder
    temporal_actual_dir = temporal_base_dir + actual_fold
    temporal_predicted_dir = temporal_base_dir+ predicted_fold
    plots_dir = temporal_base_dir  + plot_folder
    
#================================================================================================================ 
    pred_attr= 'pump_mm'
    years = set(model_input_data['year'])
    
    round_places=2
    train_x_data = pd.DataFrame()
    test_x_data = pd.DataFrame()
    train_y_data = pd.DataFrame()
    test_y_data = pd.DataFrame()
    
    selection_var = years
    selection_label = 'year'
    test_vars = test_years
    
#================================================================================================================
    for svar in selection_var :
        selected_data = model_input_data.loc[model_input_data[selection_label] == svar]
        y_t = pd.DataFrame(selected_data[pred_attr])
        x_t = selected_data
        if svar not in test_vars:
            train_x_data = train_x_data.append(x_t)
            #print(train_data_x.columns)
            train_y_data = pd.concat([train_y_data, y_t])
        else:
            test_x_data = test_x_data.append(x_t)
            test_y_data = pd.concat([test_y_data, y_t])
            
        train_x_data.to_csv(temporal_base_dir + 'train_x_data.csv', index=False)
        test_x_data.to_csv(temporal_base_dir+ 'test_x_data.csv', index=False)
        train_y_data.to_csv(temporal_base_dir + 'train_y_data.csv', index=False)
        test_y_data.to_csv(temporal_base_dir + 'test_y_data.csv', index=False)
      
    num_train_wells =  train_x_data.value_counts('PDIV_ID').count()
    score_metric_dic['# Of Training Wells'].append(num_train_wells)
    
    num_test_wells =  test_x_data.value_counts('PDIV_ID').count()
    score_metric_dic['# Of Testing Wells'].append(num_test_wells) 
    
      
    drop_columns_x = ['PDIV_ID','gridid', 'year', 'long_nad83', 'lat_nad83', 'pump_aft_acre','irr_area_mm2','pump_mm','pump_mm3']
#================================================================================================================
    train_x = train_x_data.drop(drop_columns_x ,axis=1)
    train_x.to_csv(temporal_train_dir + 'train_x.csv',index=False)
    
    train_y_mean_value = train_y_data['pump_mm'].mean()
    
    train_y_data['meanTr_mm'] = train_y_mean_value 
    train_y_mean = train_y_data['meanTr_mm']
    train_y_mean = pd.DataFrame(train_y_mean, columns =['meanTr_mm']) 
    train_y = train_y_data.drop(['meanTr_mm'],axis =1)
    train_y= train_y.iloc[:, 0].tolist()
    train_y_mean.to_csv(temporal_train_dir +  'train_y_mean.csv',index=False)
#================================================================================================================
    test_x = test_x_data.drop(drop_columns_x ,axis=1)
    test_x.to_csv(temporal_test_dir + 'test_x.csv',index=False)
    test_y_data.to_csv(temporal_test_dir + 'test_y.csv',index=False)
       
    test_y_data.columns.values[0] = 'pump_mm' 
    test_y_mean_value = test_y_data['pump_mm'].mean()
       
    test_y_data['meanTe_mm'] = test_y_mean_value 
    test_y_mean = test_y_data['meanTe_mm']
    test_y_mean = pd.DataFrame(test_y_mean, columns =['meanTe_mm'])
    test_data_y= test_y_data.drop(['meanTe_mm'],axis =1)
    test_y_data= test_y_data.iloc[:, 0].tolist()
    test_y_mean.to_csv(temporal_test_dir +  'test_y_mean.csv',index=False)
    
# #================================================================================================================     
    ranFor_model_temporal_holdout = RandomForestRegressor(random_state = 111, n_estimators = 200, max_features =5 )
    estimate = ranFor_model_temporal_holdout.fit(train_x,train_y)
    with open(temporal_model_dir + 'ranFor_model.joblib','wb') as f:
        joblib.dump(ranFor_model_temporal_holdout,f)
        
#  ================================================================================================================  
# Create ml anlaysis plots - PD and FI
        print('Plotting partial dependence .....')
        
        pdp_plots(ranFor_model_temporal_holdout,
                  train_x,
                  train_y,
                  plots_dir)
        plt.close('All')
        print('Plotting feature importance .....')
        plot_featureImportance(ranFor_model_temporal_holdout, 
                                train_x, 
                                plots_dir)
        plt.close('All')
#  ================================================================================================================  
        print('Calculating Point Scale model Train Error metrics')    
        train_predict_mm = ranFor_model_temporal_holdout.predict(train_x)
        train_predict_pt = pd.Series(train_predict_mm, name='train_predict_mm') 
        train_actual_y = train_y
        train_predict_y= train_predict_pt
      
        
        trai_pt_score = ranFor_model_temporal_holdout.score(train_x, train_y)
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
        test_predict_mm = ranFor_model_temporal_holdout.predict(test_x)
        tes_pt_score = ranFor_model_temporal_holdout.score(test_x, test_y_data)
        test_predict_pt = pd.Series(test_predict_mm, name='test_predict_mm')
     
        test_actual_y = test_y_data
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
        train_actual = train_x_data[[ 'gridid','year', 'long_nad83', 'lat_nad83','pump_mm3']]
        train_actual = train_actual.rename(columns={'pump_mm3':"actualTr_mm3" }, errors="raise")

        train_actual.to_csv(temporal_train_dir  + 'train_actual.csv',index=False)
        # train_actual - disaggregate to individual years
   
        for year in train_actual['year'].unique():
            year_data = train_actual[train_actual['year']==year]
            new_data = year_data[['gridid', 'long_nad83', 'lat_nad83','year','actualTr_mm3']].dropna(axis=0)
            new_data.to_csv(temporal_train_dir + csv_folder + 'train_actual_{}.csv'.format(year),index=False)
           
        csvs2shps(input_dir=temporal_train_dir + csv_folder, 
                  output_dir= temporal_train_dir  + shp_folder,
                  pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2)) 
#================================================================================================================   
#  Predicted training data used for development of randforest model
    train_predict_mm = ranFor_model_temporal_holdout.predict(train_x)
    train_predict= train_x_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
    train_predict['train_predict_mm']=  train_predict_mm
    
    train_predict['predTr_mm3'] =train_predict['train_predict_mm']* train_predict['irr_area_mm2']
    train_predict.to_csv(temporal_train_dir + 'train_predict.csv',index=False)
        
# train_predict - disaggregate to individual years

    for year in train_predict['year'].unique():
        year_data =train_predict[train_predict['year']==year]
        #print(year_data)
        new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','predTr_mm3']].dropna(axis=0)
        new_data.to_csv(temporal_train_dir  + csv_folder + 'train_predict_{}.csv'.format(year),index=False)

#================================================================================================================   

    train_mean_y = train_x_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
    
    train_mean_y['meanTr_mm'] = train_y_mean['meanTr_mm']
    train_mean_y['meanTr_mm3'] =train_mean_y['meanTr_mm']* train_mean_y['irr_area_mm2']
    train_mean_y.to_csv(temporal_base_dir + train_fold + 'train_mean_data_y.csv',index=False)
    
    for year in train_mean_y['year'].unique():
        year_data =train_mean_y[train_mean_y['year']==year]
        #print(year_data)
        new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','meanTr_mm3']].dropna(axis=0)
        new_data.to_csv(temporal_train_dir + csv_folder + 'train_mean_data_{}.csv'.format(year),index=False)

    
#================================================================================================================ 
    
    csvs2shps(input_dir= temporal_train_dir  + csv_folder, 
              output_dir= temporal_train_dir   + shp_folder,
              pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))  
    
    aggregate_pointValues_ingrid(input_dir=temporal_train_dir + shp_folder,
                                  shp_grid = shp_grid,ref_rast=ref_rast, 
                                  output_dir= temporal_train_dir + tiff_folder, 
                                  pattern = '*.shp') 
    
    train_grid_df= create_dataframe(input_rast_dir = temporal_train_dir + tiff_folder, 
                                      input_grid_file=shp_grid, 
                                      output_dir = temporal_train_dir + tiff_folder, 
                                      column_names=None, pattern='*.tif', 
                                      make_year_col=True, ordering=False, cellid_attr='gridid')
    
    train_grid_df.columns.values[3]  =  'wateryear'
    
    train_grid_df.to_csv(temporal_train_dir + tiff_folder + 'training.csv',index=False)
    
#================================================================================================================  
    print('Calculating Grid Scale model Train Error metrics')
    
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
    test_actual = test_x_data[['gridid', 'long_nad83', 'lat_nad83','year','pump_mm3']]
    test_actual = test_actual.rename(columns={'pump_mm3': "actualTe_mm3"}, errors="raise")

    test_actual.to_csv(temporal_test_dir  + 'test_actual.csv',index=False)
    
    # Test_Actual- disaggregate to individual years

    for year in test_actual['year'].unique():
        year_data =test_actual[test_actual['year']==year]
        #print(year_data)
        new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','actualTe_mm3']].dropna(axis=0)
        new_data.to_csv(temporal_test_dir + csv_folder +  'test_actual_{}.csv'.format(year),index=False)
#  ================================================================================================================  
#      Predicted Test data to assess model performance for unfamiliar dataset
    test_predict_mm = ranFor_model_temporal_holdout.predict(test_x)

    test_predict = test_x_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
    test_predict['predTe_mm']=  test_predict_mm
    
    test_predict['predTe_mm3'] = test_predict['irr_area_mm2']* test_predict['predTe_mm']
    
    test_predict =test_predict.drop(['irr_area_mm2','predTe_mm'],axis=1)
    
    test_predict.to_csv(temporal_test_dir  + 'test_predict.csv',index=False)
    
    
    for year in test_predict['year'].unique():
        year_data = test_predict[test_predict['year']==year]
        new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','predTe_mm3']].dropna(axis=0)
        new_data.to_csv(temporal_test_dir + csv_folder + 'test_predict_{}.csv'.format(year),index=False)
#  ================================================================================================================  
    test_mean_pump = test_x_data[[ 'gridid','long_nad83', 'lat_nad83','year','irr_area_mm2']]
    test_mean_pump['pump_mean'] = test_y_mean
    test_mean_pump['meanTe_mm3'] =test_mean_pump['pump_mean']* test_mean_pump['irr_area_mm2']
    test_mean_pump.to_csv(temporal_test_dir+ 'test_mean.csv',index=False)
    
    for year in test_mean_pump['year'].unique():
        year_data =test_mean_pump[test_mean_pump['year']==year]
        #print(year_data)
        new_data = year_data[[ 'gridid','long_nad83', 'lat_nad83','year','meanTe_mm3']].dropna(axis=0)
        new_data.to_csv(temporal_test_dir + csv_folder + 'test_mean_{}.csv'.format(year),index=False)
        
    
    csvs2shps(input_dir= temporal_test_dir  + csv_folder, 
              output_dir=temporal_test_dir  + shp_folder,
              pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))
    
    
    aggregate_pointValues_ingrid(input_dir=temporal_test_dir  + shp_folder,
                                  shp_grid = shp_grid,ref_rast=ref_rast, 
                                  output_dir= temporal_test_dir  + tiff_folder, 
                                  pattern = '*.shp') 
    
    test_grid_df = create_dataframe(input_rast_dir = temporal_test_dir  + tiff_folder, 
                                      input_grid_file=shp_grid, 
                                      output_dir = temporal_test_dir   + tiff_folder, 
                                      column_names=None, pattern='*.tif', 
                                      make_year_col=True, ordering=False, cellid_attr='cellid')
    test_grid_df.columns.values[3]  =  'wateryear' 
    test_grid_df.to_csv(temporal_test_dir + tiff_folder + 'testing.csv',index=False)
   
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
    train_actual = train_x_data.rename(columns={'pump_mm3':'actual_mm3' }, errors="raise")
    test_actual = test_x_data.rename(columns={'pump_mm3': 'actual_mm3'}, errors="raise")

    actual = pd.concat([train_actual, test_actual], axis=0)
    actual.to_csv(temporal_actual_dir + 'train_actual.csv',index=False)
    # train_actual - disaggregate to individual years

    for year in actual['year'].unique():
        year_data = actual[actual['year']==year]
        new_data = year_data[['gridid', 'long_nad83', 'lat_nad83','year','actual_mm3']].dropna(axis=0)
        new_data.to_csv(temporal_actual_dir + csv_folder + 'actual_{}.csv'.format(year),index=False)
       
    csvs2shps(input_dir=temporal_actual_dir + csv_folder, 
              output_dir=temporal_actual_dir + shp_folder,
              pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))
    
    aggregate_pointValues_ingrid(input_dir=temporal_actual_dir + shp_folder,
                                  shp_grid = shp_grid,ref_rast=ref_rast, 
                                  output_dir= temporal_actual_dir + tiff_folder, 
                                  pattern = '*.shp') 
#  ================================================================================================================  
# Creating a raster file of the predicted data
    train_predict = train_predict.rename(columns={'predTr_mm3':'predicted_mm3'}, errors="raise")
    test_predict = test_predict.rename(columns={'predTe_mm3':'predicted_mm3'}, errors="raise")
     
    predicted = pd.concat([ train_predict, test_predict], axis=0)
    
    for year in predicted['year'].unique():
        year_data = predicted[predicted['year']==year]
        new_data = year_data[['gridid', 'long_nad83', 'lat_nad83','year','predicted_mm3']].dropna(axis=0)
        new_data.to_csv( temporal_predicted_dir + csv_folder + 'predicted_{}.csv'.format(year),index=False)
    print('converting csv files to shp files')
    csvs2shps(input_dir= temporal_predicted_dir+ csv_folder, 
              output_dir= temporal_predicted_dir + shp_folder,
              pattern='*.csv', target_crs='EPSG:4326', delim=',',long_lat_pos=(1, 2))
    print('aggregating withdrawal values in a 2 km  grid cell and converting shp files to raster')
    aggregate_pointValues_ingrid(input_dir= temporal_predicted_dir + shp_folder,
                                  shp_grid = shp_grid,ref_rast=ref_rast, 
                                  output_dir=  temporal_predicted_dir + tiff_folder, 
                                  pattern = '*.shp')       
#  ================================================================================================================  
    print('Plotting actual vs predicted raster plots')
    
    
    actual_tiff_file =  temporal_actual_dir + tiff_folder  + 'actual_2017.tif'
   
  
  
    predicted_tiff_file = temporal_predicted_dir + tiff_folder  + 'predicted_2017.tif'
    
    
    temporal_actual_vs_predicted_rasterplot(model_input_file, 
                                            actual_tiff_file, 
                                            predicted_tiff_file, 
                                            test_grid_df, 
                                            plots_dir)
    
    score_metric_df = pd.DataFrame(score_metric_dic).round(round_places) 
    score_metric_df.to_csv(temporal_base_dir + 'Model_error_metrices_1219.csv',index=False)
       
    
    print(score_metric_df)
    
#  ================================================================================================================  
 
     
shp_grid = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/'\
    'for_monty/for_Monty_updates/shp_files/ref_grid.shp'
ref_rast = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/groundwater_withdrawal_prediciton/'\
    'for_monty/for_Monty_updates/ref_rasters/ref_raster_grid.tif'
model_input_file = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/topic_1/ml_input/ml_input.csv'

temporal_base_dir = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/topic_1/temporal/2012_2016/2012_2019/'
test_years = (2012,2019)

randomForest_temporal_holdout_model(model_input_file, temporal_base_dir,test_years,shp_grid,ref_rast)
