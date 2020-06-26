import class_model

class manualCheck(class_model.model):
     '''Helper Functions to test Stationarity, with ADF and/or KPSS, of the series'''

    def ADF_test (self, series):    
        
        adf = adfuller(series)
        return adf
    
        print('ADF test')
        print("\tADF Statistic: {}\n".format (adf[0])) 
        print("\tp-value: {}\n".format (adf[1]))  #Since the p-value is far from 0.05 it not stationary
        print ("\tCritical values : \n")
        for key, value in adf[4].items():
            print("\t\t{} : {}\n".format(key, value))

        if (adf[1] < 0.05):
            print('Reject the null hypothesis')
            print('Series is Stationary')
        else:
            print('Failed to reject the null hypothesis')
            print('Series is Non-Stationary')


    '''KPSS test opposite to ADF test if p>0.05 then Stationary'''    
    
    def KPSS_test (self, series):
        
        print ("\nKPSS STATISTICS\n")
        kpss_test = kpss(series, regression='c')
        return kpss_test
    
        print("\tKPSS Statistic: {}\n".format (kpss_test[0])) 
        print("\tp-value: {}\n".format (kpss_test[1]))  #Since the p-value is far from 0.05 it not stationary
        print ("\tCritical values : \n")
        for key, value in kpss_test[3].items():
            print("\t\t{} : {}\n".format(key, value))                             

        if(kpss_test[1] < 0.05):
            print('Series is Non-Stationary')
        else:
            print('Series is Stationary')
     
    def hodrick_prescott(self, series, lamb = 6.25):
        
        df = series.to_frame()
        cycle, trend = hpfilter(series, lamb= lamb)
        df['cycle'], df['trend'] = cycle, trend
        return df    
    
    '''Helper Fuctions for Seasonal Decomposition'''
    def ets_Decomposition_multiplicative(self, series):
        
        result = seasonal_decompose(series, model = 'multiplicative')
        return result
    
    def ets_Decomposition_additive(self, series):
        
        result = seasonal_decompose(series, model = 'additive')
        return result
    
