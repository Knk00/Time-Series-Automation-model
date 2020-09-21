from class_model import model

#Calculation and data manipulation related
import numpy as np
import pandas as pd

#Time Series related
from scipy.stats import kruskal
import pymannkendall as mk
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from pmdarima.arima import auto_arima 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



class automation(model):
    
    '''Replacing and detection of outliers by calculating the deviation of a datapoint from the mean of that series
        that are greater than 3 * std.
        Replacing by median value of the series'''

    def replace_outliers(series):
        
        # Calculate the absolute difference of each timepoint from the series mean
        absolute_differences_from_mean = np.abs(series - np.mean(series))

        # Calculate a mask for the differences that are > 3 standard deviations from zero
        this_mask = absolute_differences_from_mean > (np.std(series) * 2)

        # Replace these values with the median accross the data
        series[this_mask] = np.median(series)
        return series
    
     '''Determines if the series has any seasonality'''

    def seasonality_test(self, series):
        
        self.__seasonal__ = False
        idx = np.arange(len(series.index)) % 12
        H_statistic, p_value = kruskal(series, idx)
        if p_value <= 0.05:
            self.__seasonal__ = True
        return seasonal
    
     '''Helper Function to check if trend is present'''
    def mkTest(series, seasonal):
        
        if seasonal == False:
            data_mk = mk.original_test(series)
            trend = data_mk[0]
        else:
            data_mk_seasonal_test = mk.seasonal_test(series, period= 12)
            trend = data_mk_seasonal_test[0]
        
        if trend == 'decreasing' or trend == 'increasing':
            self.__trend__ = 'present' 
            trend = 'present'
            return trend
        self.__trend__ = trend
        return trend
    
     def train_test_split_(self, series):
        
        train = series[: len(series) * 0.75]
        test = series[len(series) * 0.75 :]
        return train, test
    
    '''Determines the performance of the model when tested on unbiased test data'''
    def rootMeanSquaredError(y, y_pred):
        
        rmse_ = rmse(y, y_pred)
        return rmse_
    
    #def meanAbsPercentageError(y, y_pred):
        
    #    y_true, y_pred = np.array(y_true), np.array(y_pred)
     #   return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
'''Below are all the models from which best model is chosen based on Minimum error'''

    '''Helper Functions to calculate Moving Averages'''
    def movingAverage(self, series):
        
        df = series.to_frame()
        df['SMA'] = series.rolling(window = 3).mean()
        return df

    def simpleExponentialSmoothing(self, train, test, trend):
        
        if trend == 'no trend':
            sm = SimpleExpSmoothing(train).fit()
            sm_pred = sm.forecast(len(test))
            rmse_sm = rootMeanSquaredError(test, sm_pred)
            if rmse_sm < self.rmse:
                self.rmse = rmse_sm
                self.__model__ = 'simpleExponentialSmoothing'
                
    def simpleExponentialSmoothing_forecast(self, series, forecast_range):
        sm = SimpleExpSmoothing(series).fit()
        sm_pred = sm.forecast(forecast_range)
        return sm_pred
    
    '''Helper Function Holt Winters to forecast and model seasonal data'''
    
    def holtWinters_DES(self, train, test, trend, seasonal):           
            
        if trend == 'present' and seasonal == False:

            des_add = ExponentialSmoothing(train, trend = 'add').fit().fittedvalues.shift(-1) 
            des_add_pred = des_add.forecast(len(test))
            rmse_des_add = rootMeanSquaredError(test, des_add_pred)

            des_mul =  ExponentialSmoothing(train, trend = 'mul').fit().fittedvalues.shift(-1) 
            des_mul_pred = des_mul.forecast(len(test))
            rmse_des_mul = rootMeanSquaredError(test, des_mul_pred)
            
            if rmse_des_add_ < rmse_des_mul:
                if rmse_des_add < self.rmse:
                self.rmse = rmse_des_add
                self.__model__ = 'des_add'
    

            else:
                if rmse_des_mul < self.rmse:
                self.rmse = rmse_des_add
                self.__model__ = 'des_mul'
    


    def holtWinters_DES_forecast(self, series, forecast_range, model_type):
        
        if model_type == 'add':
            des_add = ExponentialSmoothing(series, trend = 'add').fit().fittedvalues.shift(-1) 
            des_add_pred = des_add.forecast(forecast_range)
            return des_add_pred
        
        elif model_type = 'mul':
            des_mul =  ExponentialSmoothing(series, trend = 'mul').fit().fittedvalues.shift(-1) 
            des_mul_pred = des_mul.forecast(forecast_range)
            return des_mul_pred
    
            
    def holtWinters_TES(self, train, test, trend, seasonal):
        
        if: trend == 'present' and seasonal == True:
            tes_add = ExponentialSmoothing(train, trend = 'add', seasonal = 'add', seasonal_periods= 12).fit().fittedvalues
            tes_add_pred = tes_add.forecast(len(test))
            rmse_tes_add = rootMeanSquaredError(test, tes_add_pred)
            
            tes_mul = ExponentialSmoothing(train, trend = 'mul', seasonal = 'mul', seasonal_periods= 12).fit().fittedvalues
            tes_mul_pred = tes_mul.forecast(len(test))
            rmse_tes_mul = rootMeanSquaredError(test, tes_mul_pred)
            
            if rmse_tes_add < rmse_tes_mul:
                if rmse_tes_add < self.rmse:
                    self.rmse = rmse_tes_add
                    self.__model__ = 'holtWinters_TES_add'
                    self.__model_type__ = 'add'
            else:
                if rmse_tes_mul < self.rmse:
                    self.rmse = rmse_tes_mul
                    self.__model__ = 'holtWinters_TES_mul'
                    self.__model_type__ = 'mul'
                    
    def holtWinters_TES_forecast(self, series, forecast_range : int, model_type):
        
        if model_type == 'add':
            tes_add = ExponentialSmoothing(series, trend = 'add', seasonal = 'add', seasonal_periods= 12).fit().fittedvalues
            tes_add_pred = tes_add.forecast(forecast_range)
            return tes_add_pred
        
        elif model_type == 'mul':
            tes_mul = ExponentialSmoothing(series, trend = 'mul', seasonal = 'mul', seasonal_periods= 12).fit().fittedvalues
            tes_mul_pred = tes_mul.forecast(forecast_range)  
            return tes_mul_pred

    
    ''' Determines which model to choose ARIMA or SARIMA'''
    
    def arima_(self, train, test, seasonal):
        
        arima_order = auto_arima(train, seasonal= seasonal, information_criterion= 'aic')
        order = arima_order.order
        seasonal_order = arima_order.seasonal_order
        
        if seasonal_order != (0, 0, 0, 0):
            sarima = SARIMAX(train, order = order, seasonal_order = seasonal_order).fit()
            start = len(train)
            end = start + len(test) - 1
            sar_pred = sar_forecast.predict(start= start, end = end, dynamic= False, typ = 'levels')
            rmse_sarima = rootMeanSquaredError(test, sar_pred)
            self.__arimaOrder__ = order
            self.__seasonalOrder__ = seasonal_order
            
        else:
            arima_model = ARIMA(train_data, order = order)
            arima_model = arima_model.fit()
            arima_model.summary()
            start = len(train)
            end = len(train) + len(test) - 1
            arima_pred = arima_model.predict(start, end, dynamic = False, typ = 'levels')
            rmse_arima = rootMeanSquaredError(test, arima_pred)
            self.__arimaOrder__ = order
            
         
        if rmse_arima < rmse_sarima :
            if rmse_arima < self.rmse:
                self.rmse = rmse_arima
                self.__model__ = 'arima'
        else:
            if rmse_sarima < self.rmse:
                self.rmse = rmse_sarima
                self.__model__ = 'sarima'
        
    def arima_forecast(self, series, forecast_range, order, seasonal_order):
        
        if seasonal_order == (0, 0, 0, 0):
            arima_model = ARIMA(series, order = order)
            arima_model = arima_model.fit()
            arima_model.summary()
            start = len(series)
            end = start + forecast_range - 1
            arima_pred = arima_model.predict(start, end, dynamic = False, typ = 'levels')
            return arima_pred
        
        else:
            sarima = SARIMAX(series, order = order, seasonal_order = seasonal_order).fit()
            start = len(series)
            end = start + forecast_range - 1
            sar_pred = sar_forecast.predict(start= start, end = end, dynamic= False, typ = 'levels')
            return sar_pred
            
    def bestModel_forecast(self, series):
        
        if self.__model__ == 'holtWinters_TES':
            self.__forecast__ = holtWinters_TES_forecast(self.__series__, forecast_range, self.__model_type__)
            
        elif self.__model__ == 'arima' or self.__model__ == 'sarima':
            self.__forecast__ = arima_forecast(self.__series__, forecast_range, self.__arimaOrder__, self.__seasonalOrder__)
            
        elif self.__model__ == 'holtWinters_DES:
            self.__forecast__ = holtWinters_DES_forecast(self.__series__, forecast_range, self.__model_type__)
            
        elif self.__model__ == 'simpleExponentialSmoothing':
            self.__forecast__ = simpleExponentialSmoothing_forecast(self.__series__, forecast_range)
        
        return self.__forecast__
    
