# Assessment of Training Data Needs for Machine Learning Prediction of Groundwater Withdrawals 
# Abstract
Groundwater level decline threatens many aquifer systems’ sustainable supply of water for irrigated crop production. In order to implement sustainable groundwater solutions, a knowledge of groundwater withdrawals is needed. However, metering of groundwater pumping is limited in most parts of the United States and the world. Some studies have used machine learning techniques to estimate groundwater withdrawals in regions where data is abundant. Nevertheless, the data quality and quantity requirements to produce a robust estimate of groundwater withdrawals are not well studied. In many groundwater management districts of the United States, 20% or fewer of all major use wells are metered. This study sought to find an answer to the question, can a reliable groundwater withdrawal prediction machine learning model be constructed using limited data and if so, how much training data is required? in the data-rich Northwestern Kansas Groundwater Management District 4. In this study, we built point scale groundwater withdrawal prediction machine learning models using a Random Forest algorithm. The point scale prediction values are summed over a 2 km by 2 km grid. We evaluated a combination of different training splits against a constant test set to understand model performance variability. The model used public domain remote sensing, land surface model output, and hydrogeological variables as inputs and predicted withdrawals for the period from 2008 – 2020. At the 2 km scale, we observed that a model trained on 10% of the total available data showed coefficient of determination (R2) values of 0.98 and 0.75 for training and testing, respectively. The total volume of withdrawals, as well as annual variation in withdrawals, also matched observations within 91.87%. Knowledge of crop irrigation area enabled estimate aggregation over a grid, and we find that aggregation of estimates improved the spatial and temporal groundwater withdrawal estimates.  These results suggest that data-sparse regions, with 10% of all irrigation wells metered, can make reasonably accurate estimates of total groundwater withdrawals if the total irrigated area is known. This finding has significant implications for effective groundwater management in regions where there is limited data.    
# Model environment set up and steps description
The over all description of the project is summarized in [Description](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/README_DESCRIPTION.txt) and in step by step guide on how to run the Python scripts in [Steps](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/README_STEPS.txt).
The models are built using [Anaconda](https://www.anaconda.com/download) environment. The libraries used are listed in [Dependencies](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/ml_scripts/geospatial_packages.yml) file and can be installed on local computer by copying the code snippet provided below. 
```
conda env create -f  geospatial_packages.yml
```
# To cite this article
Dawit W. Asfaw, Ryan Smith, Sayantan Majumdar, Katherine Grote, V. Lakshmi, Bin Fang, J.J. Butler, Brownie Wilson. **An Assessment of Machine Learning Prediction of Groundwater Withdrawals Training Data Needs**. []()(2024). [DOI]() 
# Method workflow
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/bd0b6f3a-8d78-4303-840e-d21d0384d071)
# Model training and testing data spatial distribution map
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/fa9a75ca-9ddb-43f3-b67a-8ff92815271c)
# Model Prediction
## Point Scale Model - Actual Vs Predict time series plot 
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/d18ec2f6-26ed-4382-a3f0-04a36b66041f)


## Grid Scale Model
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/fe772a0a-01b0-48c9-81f4-3bb24b668a6e)
Observed vs predicted values within 2 km by 2 km pixel from spatial hold out model trained on 10 %. a) Actual groundwater withdrawals b) Predicted groundwater withdrawals c) Actual vs predicted scatter plot, d) Residual of predicted values vs residual.

# Affiliations
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/6c5743ea-a83e-4f83-8363-eb0585ee0b72)

# Funding Agency
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/3125ed01-e1c8-416e-b4a6-131febe2f056)








