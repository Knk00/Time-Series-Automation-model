# Time-Series-Automation-model #
This repo contains methods and automation that I implmented for a Demand Forcasting project. The goal of the project was to forecast the Sales. For this, I developed an 
Automation  model that forecasts based on a given time series. The model consists of models like ARIMA, SARIMA, Holt-Winters and others.

## Framework of the Automation model ##
The model is divided into two parts. When designing a model, there is a possibility that there might be a bug. To handle this, I have included a Manual-Check Functionality.

The Manual-Check Funcitonality includes Statistical tests and other visualization techniques that can provide valuable insight on the series. The Functionality includes:
    
    * ETS Decomposition : Divides the series into its principal components i.e. Trend, Seasonality and Residual.
    * Hodrick-Prescott Filter : Divides the series into Trend and Cycle.
    * ADF and KPSS test : Stationarity check.
    
Right now, these are the fucntionalities that have been added but, I will be adding more functionalities to this.

