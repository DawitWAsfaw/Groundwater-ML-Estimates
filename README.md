# Predicting Groundwater Withdrawals Using Machine Learning with Limited Metering Data: Assessment of Training Data Requirements 
# Abstract
Groundwater level declines threaten the long-term prospects of many aquifers supporting irrigated agriculture. In order to implement sustainable groundwater solutions for these systems, a time series of groundwater pumping is needed. However, metering of pumping is limited in most parts of the United States and elsewhere. Some studies have used machine learning techniques to estimate pumping in regions where metering data are abundant. However, the data quality and quantity requirements to produce a robust estimate of regional groundwater pumping are not readily available or well-studied. In many areas of the United States, 20% or fewer of high-capacity wells are metered. This study seeks to determine which parameters are most useful for predicting groundwater pumping and what quantity of data is needed. We carried out this study in a data-rich groundwater management district in the High Plains aquifer in the state of Kansas in the central United States. We built pumping prediction machine learning models using a random forest algorithm that was based on public domain remote sensing data, land surface model output, and hydrogeological variables to predict pumping for the period from 2008 – 2020. We predicted pumping at two spatial scales, point scale (individual wells) and over a 2 km by 2 km grid where data are aggregated within each grid cell. For both scales of prediction, we evaluated a combination of different training splits against a constant test set to understand the performance variability of the models. Predictions based on point-scale inputs did not sufficiently capture the variability of actual pumping measurements. But at the 2 km scale, we observed that a model trained on 10% of the total available data showed coefficient of determination (R2) values of 0.98 and 0.75 for training and testing, respectively. The total predicted volume of pumping, as well as annual variation in pumping, also matched observations within 3%. Knowledge of crop irrigation area enabled summing up predicted pumping over a grid and also reduced uncertainty of pairing individual wells to irrigated areas by aggregating spatially, and we find that summing up of estimates improved the spatial and temporal pumping estimates. These results suggest that in data-sparse regions, if 10% of all irrigation wells are metered, reasonably accurate estimates of regional irrigation pumping are possible at the 2 km by 2 km scale if the irrigated area is known. This finding has significant implications for groundwater management in regions where metering is limited.   
# Model environment set up and steps description
The over all description of the project is summarized in [Description](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/README_DESCRIPTION.txt) and in step by step guide on how to run the Python scripts in [Steps](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/README_STEPS.txt).
The models are built using [Anaconda](https://www.anaconda.com/download) environment. The libraries used are listed in [Dependencies](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/ml_scripts/geospatial_packages.yml) file and can be installed on local computer by copying the code snippet provided below. 
```
conda env create -f  geospatial.yml
```
# To cite this article
Asfaw, Dawit Wolday and Smith, Ryan and Majumdar, Sayantan and Grote, Katherine and Fang, Bin and Wilson, Blake B. and Lakshmi, Venkataraman and Butler, James J., Predicting Groundwater Withdrawals Using Machine Learning with Limited Metering Data: Assessment of Training Data Requirements. Available at SSRN: https://ssrn.com/abstract=5177058 or http://dx.doi.org/10.2139/ssrn.5177058 
# Method workflow
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/bd0b6f3a-8d78-4303-840e-d21d0384d071)
# Model training and testing data spatial distribution map
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/fa9a75ca-9ddb-43f3-b67a-8ff92815271c)
# Model Prediction
## Point Scale Model - Actual Vs Predict time series plot 
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/46729ffa-9fa9-426b-a064-2a91a69deda8)
Point scale time series data. Annual mean actual vs predicted time series plot for models trained on 90% (a), 80% (b), 60% (c), 40% (d), 20% (e), and 10% (f). Red lines are predicted values and blue lines show observed groundwater withdrawal data.

## Grid Scale Model
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/7083c392-730e-4c62-915f-1e6d25e188c7)
Observed vs predicted values within 2 km by 2 km pixel from spatial hold out model trained on 10%. a) Actual groundwater withdrawals b) Predicted groundwater withdrawals c) Actual vs predicted scatter plot, d) Residual of predicted values vs residual.

# Affiliations
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/6c5743ea-a83e-4f83-8363-eb0585ee0b72)

# Funding Agency
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/3125ed01-e1c8-416e-b4a6-131febe2f056)








