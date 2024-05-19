import os

# ======================================================================================================
def create_spatio_temporal_estimates(spatio_temporal_estimates):
    '''
    Creates a folder for individual fraction used to build a model
    ----------
    :Param base_dir : main directory path string under which individual sub-folders for each percentage will be created
    :return: None

    '''
    base_dir = spatio_temporal_estimates 
    for number in range(0,95, 10):
        number = str(number)
        folder_name =  number + "_percent"
        folder_path = os.path.join(base_dir, folder_name)
        # print(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        for folder in os.listdir(base_dir):
            #print(folder)
            
            folder_name1 = folder + '/train'
            folder_name2 = folder + '/test'
           
            folder_name3 = folder + '/actual'
            folder_name4 = folder + '/predicted'
   
            folder_path1 = os.path.join(base_dir, folder_name1)
            folder_path2 = os.path.join(base_dir, folder_name2)
            folder_path3 = os.path.join(base_dir, folder_name3)
            folder_path4 = os.path.join(base_dir, folder_name4)

            os.makedirs(folder_path1, exist_ok=True)
            os.makedirs(folder_path2, exist_ok=True)
            os.makedirs(folder_path3, exist_ok=True)
            os.makedirs(folder_path4, exist_ok=True)
 
            sub_path = os.path.join(base_dir, folder)
            sub_path1 = str(sub_path) + '/'
           
           
            for sub_folders in os.listdir(sub_path1):
                sub_folder1 = sub_folders + '/shp'
                sub_folder2 = sub_folders + '/csv'
                sub_folder3 = sub_folders  + '/tiff'
               
                sub_folder_path1 = os.path.join(sub_path1, sub_folder1)
                sub_folder_path2 = os.path.join(sub_path1, sub_folder2)
                sub_folder_path3 = os.path.join(sub_path1, sub_folder3)
               
                os.makedirs(sub_folder_path1, exist_ok=True)
                os.makedirs(sub_folder_path2, exist_ok=True)
                os.makedirs(sub_folder_path3, exist_ok=True)
    
    for number in range(10,95, 10):
        number = str(number)
        folder_name =  number + "_percent"
        folder_path = os.path.join(base_dir, folder_name)
        for folder in os.listdir(base_dir):
            #print(folder)
           
            folder_name1 = folder + '/model'
            folder_name2 = folder + '/plots'
   
            folder_path1 = os.path.join(base_dir, folder_name1)
            folder_path2 = os.path.join(base_dir, folder_name2)

            os.makedirs(folder_path1, exist_ok=True)
            os.makedirs(folder_path2, exist_ok=True)
          
# #======================================================================================================         
def create_temporal_estimate_dir(temporal_estimate_dir):
    '''
    Create a folder for temporar estimate files
    ----------
    : param base_yearly_dir: parent directory where temporal estimate analysis results is saved
    : return: None

    '''
    folder_name1 = temporal_estimate_dir + '/test/'
    folder_name2 =  temporal_estimate_dir + '/train/'
 
    folder_name3 = temporal_estimate_dir  + '/actual'
    folder_name4 = temporal_estimate_dir + '/predicted'

    folder_path1 = os.path.join(temporal_estimate_dir, folder_name1)
    folder_path2 = os.path.join(temporal_estimate_dir, folder_name2)
    folder_path3 = os.path.join(temporal_estimate_dir, folder_name3)
    folder_path4 = os.path.join(temporal_estimate_dir, folder_name4)

    os.makedirs(folder_path1, exist_ok=True)
    os.makedirs(folder_path2, exist_ok=True)
    os.makedirs(folder_path3, exist_ok=True)
    os.makedirs(folder_path4, exist_ok=True)
    
    sub_path = os.path.join(temporal_estimate_dir )

    for sub_folders in os.listdir(sub_path):
        sub_folder1 = sub_folders + '/shp'
        sub_folder2 = sub_folders + '/csv'
        sub_folder3 = sub_folders  + '/tiff'
        
        sub_folder_path1 = os.path.join(sub_path, sub_folder1)
        sub_folder_path2 = os.path.join(sub_path, sub_folder2)
        sub_folder_path3 = os.path.join(sub_path, sub_folder3)
        
        os.makedirs(sub_folder_path1, exist_ok=True)
        os.makedirs(sub_folder_path2, exist_ok=True)
        os.makedirs(sub_folder_path3, exist_ok=True)
    
  
    folder_name3 = temporal_estimate_dir  + '/model'
    folder_name4 =  temporal_estimate_dir  + '/plots'
    
    folder_path3 = os.path.join(temporal_estimate_dir , folder_name3)
    folder_path4 = os.path.join(temporal_estimate_dir , folder_name4) 
    os.makedirs(folder_path3, exist_ok=True) 
    os.makedirs(folder_path4, exist_ok=True)  
 
#======================================================================================================
base_dir = 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/topic_1/spatial/5_17_2024/'
temporal_estimate_dir= 'C:/Users/dasfaw/OneDrive - Colostate/Documents/Spring2024/Research/topic_1/temporal/2012_2016/2012_2019/'
#======================================================================================================
create_spatio_temporal_estimates(base_dir)  
# create_temporal_estimate_dir(temporal_estimate_dir) 