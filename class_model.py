class model:
    
    '''Attributes of the class are the same as the ones the describe any time series : Trend, Seasonality; additionally
    Root-Mean-Squared Error and Model type (additive/multiplicative) are also taken into consideration'''
    
    def __init__(self, series, rmse = None, model = None):
        
        if rmse == None:
            self.rmse : float = float('inf')
        else:
            self.rmse = rmse
        self.__model__ = model
        self.__seasonal__ = None
        self.__trend__ = None
        self.__forecast__ = None
        self.__model_type__ = None
        self.__series__ = series
