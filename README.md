# An Assessment of Machine Learning Prediction of Groundwater Withdrawals Training Data Needs
# Abstract
Groundwater level declines threaten many aquifer systems’ sustainable supply of water for irrigated crop production. In order to implement sustainable groundwater solutions, a knowledge of groundwater withdrawals is needed. However, metering of groundwater pumping is limited in most parts of the United States and world. Some studies have used machine learning techniques to estimate groundwater withdrawals in regions where data is abundant. Nevertheless, the data quality and quantity requirements to produce a robust estimate of groundwater withdrawals are not well studied. In many groundwater management districts of the United States, 20% or fewer of all major use wells are metered. This study sought to estimate the potential performance of machine learning models in such data-poor regions by evaluating multiple splits of training and testing data in the data-rich Northwestern Kansas Groundwater Management District 4. In this study, we built point scale groundwater withdrawal prediction machine learning models using a Random Forest algorithm. The point scale prediction values are summed over a 2 km by 2 km grid. We evaluated a combination of different training splits against a constant test set to understand model performance variability. The model used public domain remote sensing, land surface model output, and hydrogeological variables as inputs and predicted withdrawals for the period from 2008 – 2020. At the 2 km scale, we observed that a model trained on 10 % of the total available data showed coefficient of determination (R2) values of 0.98 and 0.75 for training and testing, respectively. The total volumes of withdrawals, as well as annual variation in withdrawals, also matched observations within 91.87 %  . Knowledge of crop irrigation area enabled estimate aggregation over a grid, and we find that aggregation of estimates improved the spatial and temporal groundwater withdrawal estimates.  These results suggest that data-sparse regions, with 10% of all irrigation wells metered, can make reasonably accurate estimates of total groundwater withdrawals if the total irrigated area is known. This finding has significant implications for effective groundwater management in regions where there is limited data.  
# Model environment set up
The models are built using [Anaconda](https://www.anaconda.com/download) environment. The libraries used are listed in geospatial_packages.yml file. 
```
conda env create -f  geospatial_packages.yml
```

# Method workflow
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/bd0b6f3a-8d78-4303-840e-d21d0384d071)
# Model training and testing data spatial distribution map
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/fa9a75ca-9ddb-43f3-b67a-8ff92815271c)
# Model Prediction
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/fe772a0a-01b0-48c9-81f4-3bb24b668a6e)
Observed vs predicted values within 2 km by 2 km pixel from spatial hold out model trained on 10 %. a) Actual groundwater withdrawals b) Predicted groundwater withdrawals c) Actual vs predicted scatter plot, d) Residual of predicted values vs residual.

# Affiliations
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/e61675f5-231f-4899-afcd-35e569bb4367)

# Funding Agency
![image](https://github.com/DawitWAsfaw/Groundwater-ML-Estimates/assets/89609490/3125ed01-e1c8-416e-b4a6-131febe2f056)








