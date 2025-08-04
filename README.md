# Predicting Groundwater Withdrawals Using Machine Learning with Limited Metering Data: Assessment of Training Data Requirements 
# Abstract
The future of major aquifer systems supporting irrigated agriculture is threatened due to unsustainable groundwater pumping. Metering of pumping is key for implementing robust groundwater management, but metering is limited in most aquifers. Although machine learning methods have been used to estimate pumping over certain regions, these studies have not fully demonstrated the data quantity and input parameter requirements to accurately estimate regional groundwater pumping. This study determined the data quantity required and identified relevant features to develop Random Forests-based annual groundwater pumping estimates (2008–2020) over the Kansas High Plains aquifer. We predicted pumping at two spatial scales, i.e., point (well) and grid (2 km). We evaluated a combination of different training splits against a constant test set to understand the performance of the models. Summing predicted pumping over a 2 km grid was made possible with knowledge of crop irrigation area. This knowledge also decreased the uncertainty observed in linking individual wells with irrigated areas and further improved the spatial and temporal pumping estimates. At the 2 km scale, we observed that a model trained on 10 % of the total available data had coefficient of determination (R2) values of 0.98 and 0.75 for training and testing, respectively. These results show reasonable estimates of irrigation pumping are possible at the 2 km scale when 10 % of irrigation wells are metered and if the irrigated area is known. This finding has significant implications for groundwater management in many heavily stressed aquifers.
# Model environment set up and steps description
The over all description of the project is summarized in [Description](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/README_DESCRIPTION.txt) and in step by step guide on how to run the Python scripts in [Steps](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/README_STEPS.txt).
The models are built using [Anaconda](https://www.anaconda.com/download) environment. The libraries used are listed in [Dependencies](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/blob/main/ml_scripts/geospatial_packages.yml) file and can be installed on local computer by copying the code snippet provided below. 
```
conda env create -f  geospatial.yml
```
# To cite this article
Asfaw, Dawit Wolday and Smith, Ryan and Majumdar, Sayantan and Grote, Katherine and Fang, Bin and Wilson, Blake B. and Lakshmi, Venkataraman and Butler, James J., Predicting Groundwater Withdrawals Using Machine Learning with Limited Metering Data: Assessment of Training Data Requirements. Available at [SSRN: https://ssrn.com/abstract=5177058 or http://dx.doi.org/10.2139/ssrn.5177058 ](https://www.sciencedirect.com/science/article/pii/S0378377425004056)
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








