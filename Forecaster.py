import pandas as pd
import numpy as np
import os
import pandas_datareader as pdr
from collections import Counter
from scipy.stats import pearsonr
import rpy2.robjects as ro

rwd = os.getcwd().replace('\\','/')

class Forecaster:
    def __init__(self,name=None,y=None,current_dates=None,future_dates=None,
                 current_xreg=None,future_xreg=None,forecast_out_periods=24):
    """
    """
    self.name = name
        self.y = y
        self.current_dates = current_dates
        self.future_dates = future_dates
        self.current_xreg = current_xreg
        self.future_xreg = future_xreg
        self.ordered_xreg = None
        self.forecast_out_periods=forecast_out_periods
        self.info = {}
        self.mape = {}
        self.forecasts = {}
        self.feature_importance = {}
        self.best_model = ''

    def get_data_fred(self):
        pass

    def process_xreg_df(self):
        pass

    def forecast_arima(self):
        pass

    def forecast_mlp(self):
        pass

    def set_best_model(self):
        pass

