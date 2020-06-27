# Time-Series-Automation-model #
This repo contains methods and automation that I implmented for a Demand Forcasting project. The goal of the project was to forecast the Sales. For this, I developed an 
Automation  model that forecasts based on a given time series. The model consists of models like ARIMA, SARIMA, Holt-Winters and others.

## Framework of the Automation model ##
When designing a model, there is a possibility that there might be a bug. Therefore, the model is divided into two parts. Automated and Manual-Check.

The Manual-Check Funcitonality includes Statistical tests and other visualization techniques that can provide valuable insight on the series. The Functionality includes:
    
    * ETS Decomposition : Divides the series into its principal components i.e. Trend, Seasonality and Residual.
    * Hodrick-Prescott Filter : Divides the series into Trend and Cycle.
    * ADF and KPSS test : Stationarity check.
    
Right now, these are the fucntionalities that have been added but, I will be adding more functionalities to this.

The Automation functionality includes all Non-Parametric tests, Error Metrics, Data Cleaning methods and Algorithms to return the forecast of the series. The Automation part includes:

    * Kruskal-Wallis test : Determines if the series is seasonal
    * Mann-Kendall test : Determines if the series has trend (non-seasonal/seasonal trend)
    * Replace Outliers using the median of the series
    * Train-Test Split
    * Algorithms :
        * Simple Exponential Smoothing
        * ARIMA
        * Seasonal ARIMA
        * LSTM 
        * Holt's Linear trend method (additve/multiplicative)
        * Holt-Winters (additve/multiplicative)
    * Root-Mean Squared Error to choose the best fit algorithm for the series.
    
This is not the end! I know there are many more functionalities that can be added. So, do add anything or suggest if you deem fit. I will also be updating this Repo from time to time as I learn more fun and exciting techniques !!!

