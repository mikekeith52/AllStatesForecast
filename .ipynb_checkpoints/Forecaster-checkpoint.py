import pandas as pd
import numpy as np
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import rpy2

rwd = os.getcwd().replace('\\','/')

class Data:
    def __init__(self,xreg=None):
        """ Parameters: xreg : dict, default None
                            a dictionary of external regressors
                            keys are the regressors' names
                            values are a list of regressor values
                            must be the same length as self.y
            Attributes: 
        """
        self.name = ''
        self.y = []
        self.y_dates = []
        self.xreg = xreg
        self.info = {}
        self.mape = {}
        self.forecasts = {}

    def get_data(self,series,date_start=None,date_end=None):
        """ 
        """
        try:
            importr('quantmod')
        except:
            ro.r("install.packages('quantmod')")
            importr('quantmod')
        
        ro.r(f"""
            symbols = '{series}'
            options("getSymbols.warning4.0"=FALSE)
            getSymbols(symbols,src='FRED',auto.assign=T)

            df <- data.frame({series})
            write.csv(df,'{rwd}/tmp/tmp.csv',row.names=T)
        """)
        
        tmp = pd.read_csv('tmp/tmp.csv',index_col=0)
        self.name = series
        self.y = list(tmp[f'{series}'])
        self.y_dates = list(tmp.index)

    def forecast_arima(self):
        pass

    def forecast_tbats(self):
        pass

    def forecast_rf(self):
        pass