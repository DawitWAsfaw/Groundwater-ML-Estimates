from sklearn.inspection import PartialDependenceDisplay 
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
import seaborn as sns
import rasterio as rio


plt.rcParams["font.weight"] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['font.family'] = ['arial']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 20})


plt.close('all')

def pdp_plots(estimate,train_x,train_y,plots_dir):
    """
    Create partial dependence plot(PD) 
    : estimate: random forest model used for prediction
    : train_x: predictor training data set used to build random forest model
    : train_y: target training data set used to test random forest model during model training
    : plots_dir : directory path to store  plots 
    : return: none
    """
   
 
    # print(train_x.columns)
    renaming_dic = {'gwlchan':'GW level change [m]',
                    'gwl': 'GW level [m]',
                    'tmax':'Temperature (max) [°C]',
                    'slope':'Slope',
                    'ppt':'Precipitation [mm]',
                    'evap':'Evapotranspiration [mm]',
                    'aquiThick':'Aquifer Thickness [m]',
                    'tmin':'Temperature (min) [°C]',
                    'cropwaterdem':'Crop water demand [mm]'
                    }
    
    # print(train_x.columns)
    train_x.rename(columns=renaming_dic, inplace=True)
    
   

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)


    fig1.supylabel('Groundwater withdrawals [mm]')
    feature_names = {
        "feature1": ['Precipitation [mm]'],
        "feature2": ['Evapotranspiration [mm]'],
        "feature3": ['GW level change [m]'],
        "feature4": ['Aquifer Thickness [m]'],
        "kind": "average",
    } 
        
    d1 = PartialDependenceDisplay.from_estimator(estimate,train_x,feature_names["feature1"],  ax=ax1)
    d1.axes_[0][0].set_ylabel(' ')
    d1.axes_[0][0].set_ylim(200,375)
    d1.axes_[0][0].annotate('a) ',xy=(200, 300),fontsize="20")
    d2 = PartialDependenceDisplay.from_estimator(estimate,train_x,feature_names["feature2"],  ax=ax2)  
    d2.axes_[0][0].set_ylabel(' ')
    d2.axes_[0][0].set_ylim(200,375)
    d2.axes_[0][0].annotate('b) ',xy=(29.8, 312),fontsize="20")
    d3 = PartialDependenceDisplay.from_estimator(estimate,train_x,feature_names["feature3"],  ax=ax3)
    d3.axes_[0][0].set_ylabel(' ')
    d3.axes_[0][0].set_ylim(200,375)
    d3.axes_[0][0].annotate('c) ',xy=(600, 350),fontsize="20")
    d4 = PartialDependenceDisplay.from_estimator(estimate,train_x,feature_names["feature4"],  ax=ax4) 
    d4.axes_[0][0].set_ylabel(' ')
    d4.axes_[0][0].set_ylim(200,375)
    d4.axes_[0][0].annotate('d) ',xy=(370, 295),fontsize="20")
    

    fig1.savefig(plots_dir + 'Climatic_PDP.jpg',dpi=300)



#===========================================================================================================

def plot_featureImportance(ranFor_model_optimized, train_x, plots_dir):
    """
    Create feature importance plot (FI)
    : ranFor_model_optimized: random forest model used for prediction
    : train_x: predictor training data set used to build random forest model
    : plots_dir : directory path to store  plots 
    : return: none
    """
    # print(train_x.columns)
    predictor_features_dict = {'GW level change [m]':'GWLC [m]',
                    'GW level [m]': 'GWL [m]',
                    'Temperature (max) [°C]':'T(max) [°C]',
                    'Precipitation [mm]':'P [mm]',
                    'Slope': 'S ',
                    'Evapotranspiration [mm]':'ET [mm]',
                    'Aquifer Thickness [m]':'AT[m]',
                    'Temperature (min) [°C]':'T(min) [°C]',
                    'Crop water demand [mm]':'CWD [mm]'
                    }
    
    predictor_features_df = train_x.rename(columns=predictor_features_dict)
    # print(predictor_features_df.columns)
    labels_x_axis= np.array(predictor_features_df .columns)
    importance = np.array(ranFor_model_optimized.feature_importances_)
    imp_dict = {'feature_names': labels_x_axis, 'Feature_importance': importance}
    imp_df = pd.DataFrame(imp_dict)
    imp_df.sort_values(by=['Feature_importance'], ascending=False, inplace=True)
    plt.figure(figsize=(20, 8))
    plt.rcParams['font.size'] = 20
    sns.barplot(x=imp_df['feature_names'], y=imp_df['Feature_importance'], color ='gray')
    plt.xticks(rotation=0)
    plt.ylabel('Gini Importance')
    plt.xlabel(' ')
    plt.tight_layout()
    plt.savefig(( plots_dir +  'pred_importance.png'), dpi=600)
    print('Feature importance plot saved')


# ========================================================================================================================
def spatio_temporal_actual_vs_predicted_rasterplot(model_input_file, actual_tiff_file, predicted_tiff_file, test_grid_df, plots_dir):
    """
    Create raster, scatter plots for actual vs predicted values and scatter plot predicted residuals vs predicted values
    : ranFor_model_optimized: spatial temporal prediction random forest model used for prediction
    : train_x: predictor training data set used to build random forest model
    : plots_dir : directory path to store  plots 
    : return: none
    """
     
    df_file = pd.read_csv(model_input_file)
    long = df_file['long_nad83'].to_numpy()
    lat = df_file['lat_nad83'].to_numpy()
    min_long = min(long)
    max_long = max(long) 
    min_lat = min(lat)
    max_lat = max(lat)
    
    fig, axs = plt.subplots(2,2,squeeze=False,figsize=(15,7),layout="constrained")  
    actual_tiff = rio.open(actual_tiff_file)
    actual_tiff_matrix= actual_tiff.read(1)
    actual_tiff_matrix= ma.masked_greater(actual_tiff_matrix , 300,copy=True)
    predicted_tiff = rio.open(predicted_tiff_file)
    predicted_tiff_matrix = predicted_tiff.read(1)
    predicted_tiff_matrix  = ma.masked_greater(predicted_tiff_matrix, 300,copy=True)

    cax = axs[0,0].imshow(actual_tiff_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'coolwarm',aspect='auto')
    cax1 = axs[0,0].inset_axes([0.35, 0.92, 0.6, 0.03])
    ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
    ab.ax.set_ylabel('mm')
    axs[0,0].set_xscale('linear')
    axs[0,0].set_xlabel('Longitude (Degree)')
    axs[0,0].set_ylabel('Latitude (Degree)')
    axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="20")
    

    b, a = np.polyfit(test_grid_df[test_grid_df.columns[0]],  test_grid_df[test_grid_df.columns[2]], deg=1)
    axs[0,1].plot(test_grid_df[test_grid_df.columns[0]], a + b *  test_grid_df[test_grid_df.columns[0]], color="black",lw=1);

    axs[0,1].scatter(test_grid_df[test_grid_df.columns[0]],  test_grid_df[test_grid_df.columns[2]], linewidth=1,color ='brown',label='Actual',alpha =0.25)
    axs[0,1].set_facecolor("white")

    axs[0,1].set_xlabel('Actual (mm)')
    axs[0,1].set_ylabel('Predicted (mm)')
    axs[0,1].annotate('b) ',xy=(400,0),fontsize="20")
    
    
    cax = axs[1,0].imshow(predicted_tiff_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'coolwarm',aspect='auto')
    cax2 = axs[1,0].inset_axes([0.35, 0.92, 0.6, 0.03])
    ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
    ab.ax.set_ylabel('mm')
    axs[1,0].set_facecolor("white")
    axs[1,0].set_xscale('linear')
    axs[1,0].set_xlabel('Longitude (Degree)')
    axs[1,0].set_ylabel('Latitude (Degree)')
    axs[1,0].annotate('c)',xy=(-100, 39.13),fontsize="20")
    
   
    
    
    renaming_dic = {test_grid_df.columns[0]:'Actual (mm)',
                    test_grid_df.columns[2]:'Predicted (mm)',
                  }

    test_grid_df['Residuals (mm)'] = test_grid_df[test_grid_df.columns[0]]- test_grid_df[test_grid_df.columns[2]]
    test_grid_df.rename(columns=renaming_dic, inplace=True)


    axs[1,1].axhline(color ='r',linestyle = '--',linewidth=1.5)

    axs[1,1].scatter(test_grid_df['Predicted (mm)'],test_grid_df['Residuals (mm)'], linewidth=1,color ='gray',label='Actual',alpha =0.05)
    axs[1,1].set_facecolor("white")

    axs[1,1].set_xlabel('Predicted (mm)')
    axs[1,1].set_ylabel('Residuals (mm)')
    axs[1,1].annotate('d)',xy=(400,-150),fontsize="20")
    
    # plt.tight_layout()
    plt.savefig((plots_dir +   'spatio_temporal_actual_observed_raster_plot.png'), dpi=600)
    
    plt.close('all')
    

# ========================================================================================================================

def temporal_actual_vs_predicted_rasterplot(model_input_file, actual_tiff_file, predicted_tiff_file, test_grid_df, plots_dir):
   """
   Create raster, scatter plots for actual vs predicted values and scatter plot predicted residuals vs predicted values
   : ranFor_model_optimized: temporal prediction random forest model used for prediction
   : train_x: predictor training data set used to build random forest model
   : plots_dir : directory path to store the plots 
   : return: none
   """   
   df_file = pd.read_csv(model_input_file)
   long = df_file['long_nad83'].to_numpy()
   lat = df_file['lat_nad83'].to_numpy()
   min_long = min(long)
   max_long = max(long) 
   min_lat = min(lat)
   max_lat = max(lat)
     
   fig, axs = plt.subplots(2,2,squeeze=False,figsize=(15,7),layout="constrained")
   actual_tiff = rio.open(actual_tiff_file)
   actual_tiff_matrix= actual_tiff.read(1)
   actual_tiff_matrix= ma.masked_greater(actual_tiff_matrix , 300,copy=True)
   predicted_tiff = rio.open(predicted_tiff_file)
   predicted_tiff_matrix = predicted_tiff.read(1)
   predicted_tiff_matrix  = ma.masked_greater(predicted_tiff_matrix, 300,copy=True)
        

   cax = axs[0,0].imshow(actual_tiff_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'hsv',aspect='auto')
   cax1 = axs[0,0].inset_axes([0.35, 0.92, 0.6, 0.03])
   ab = fig.colorbar(cax,cax=cax1,orientation='horizontal')
   ab.ax.set_ylabel('mm')
   axs[0,0].set_xscale('linear')
   axs[0,0].set_xlabel('Longitude (Degree)')
   axs[0,0].set_ylabel('Latitude (Degree)')
   axs[0,0].annotate('a)',xy=(-100, 39.13),fontsize="20")
   
   b, a = np.polyfit(test_grid_df['test_actual'],  test_grid_df['test_predict'], deg=1)
   axs[0,1].plot( test_grid_df['test_actual'], a + b *  test_grid_df['test_actual'], color="yellow",lw=1);
    
   axs[0,1].scatter( test_grid_df['test_actual'], test_grid_df['test_predict'], linewidth=1,color ='blue',label='Actual',alpha =0.25)
   axs[0,1].set_facecolor("white")
    
   axs[0,1].set_xlabel('Actual (mm)')
   axs[0,1].set_ylabel('Predicted (mm)')
   axs[0,1].annotate('b) ',xy=(400,0),fontsize="20")
   
    
   cax = axs[1,0].imshow(predicted_tiff_matrix , extent=(min_long,max_long, min_lat,max_lat),origin='upper',cmap = 'hsv',aspect='auto')
   cax2 = axs[1,0].inset_axes([0.35, 0.92, 0.6, 0.03])
   ab = fig.colorbar(cax,cax=cax2,orientation='horizontal')
   ab.ax.set_ylabel('mm')
   axs[1,0].set_facecolor("white")
   axs[1,0].set_xscale('linear')
   axs[1,0].set_xlabel('Longitude (Degree)')
   axs[1,0].set_ylabel('Latitude (Degree)')
   axs[1,0].annotate('c)',xy=(-100, 39.13),fontsize="20")
    
    
   renaming_dic = {'test_actual':'Actual (mm)',
                'test_predict':'Predicted (mm)',
              }
    
   test_grid_df['Residuals (mm)'] = test_grid_df['test_actual']- test_grid_df['test_predict']
   test_grid_df.rename(columns=renaming_dic, inplace=True)
    
    
   axs[1,1].axhline(color ='r',linestyle = '--',linewidth=1.5)
    
   axs[1,1].scatter(test_grid_df['Predicted (mm)'],test_grid_df['Residuals (mm)'], linewidth=1,color ='black',label='Actual',alpha =0.05)
   axs[1,1].set_facecolor("white")
    
   axs[1,1].set_xlabel('Predicted (mm)')
   axs[1,1].set_ylabel('Residuals (mm)')
   axs[1,1].annotate('d)',xy=(400,-150),fontsize="20")
    
   # plt.tight_layout()
   plt.savefig((plots_dir +   'temporal_actual_observed_raster_plot.png'), dpi=600)
   plt.close('all')
    
    
   plt.rcdefaults()
    
    
def actual_vs_predicted_timeseries_plot(test_yearly_aver, base_dir):
    """
    Create time series plots for actual vs predicted values for models built using 10%, 20%, 40%, 60%, 80%, 90% training data
    : test_yearly_aver: dataframe holding actual and predicted values for all 17 models
    : plots_dir : directory path to store the plots 
    : return: none
    """
   
    plt.close('all')
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2,squeeze=False,figsize=(15,12))
    fig.supylabel('Groundwater withdrawals [mm]')
    fig.supxlabel('Year')
    
    # ===============================================================================================================================================
    ax1.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax1.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_10P'], linewidth=1,color ='red',label='Predicted')

    ax1.annotate('a)',xy=(2008, 47),fontsize="20")
    ax1.legend(loc='upper right',fontsize="20",frameon =False)
    ax1.set_facecolor("white")
    

    # ===============================================================================================================================================

    ax2.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax2.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_20P'], linewidth=1,color ='red',label='Predicted')

    ax2.annotate('b)',xy=(2008, 47),fontsize="20")
    ax2.legend(loc='upper right',fontsize="20",frameon =False)
    ax2.set_facecolor("white")

    # ===============================================================================================================================================
    ax3.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax3.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_40P'], linewidth=1,color ='red',label='Predicted')

    ax3.annotate('c)',xy=(2008, 47),fontsize="20")
    ax3.legend(loc='upper right',fontsize="20",frameon =False)
    ax3.set_facecolor("white")
  
    # ===============================================================================================================================================
    ax4.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax4.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_60P'], linewidth=1,color ='red',label='Predicted')


    ax4.legend(loc='upper right',fontsize="20",frameon =False)
    ax4.annotate('d)',xy=(2008, 47),fontsize="20")
    ax4.set_facecolor("white")


    # ===============================================================================================================================================
    ax5.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax5.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_80P'], linewidth=1,color ='red',label='Predicted')

    ax5.annotate('e)',xy=(2008, 47),fontsize="20")
    ax5.legend(loc='upper right',fontsize="20",frameon =False)
    ax5.set_facecolor("white")
   

    # ===============================================================================================================================================
    ax6.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_actual'], linewidth=1,color ='blue',label='Actual')
    ax6.plot(test_yearly_aver['wateryear'],test_yearly_aver['test_predict_90P'], linewidth=1,color ='red',label='Predicted')

    ax6.annotate('f)',xy=(2008, 47),fontsize="20")
    ax6.legend(loc='upper right',fontsize="20",frameon =False)
    ax6.set_facecolor("white")

    
    plt.tight_layout()
    plt.savefig((base_dir  +   'actual_observed_timeseries_plot.png'), dpi=600)
