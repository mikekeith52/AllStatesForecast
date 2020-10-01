import pandas as pd
import numpy as np
import os
import pandas_datareader as pdr

rwd = os.getcwd().replace('\\','/')

class Forecaster:
    def __init__(self,name=None,y=None,current_dates=None,future_dates=None,
                 current_xreg=None,future_xreg=None,forecast_out_periods=27):
        """ Parameters: name : str
                        y : list
                        curren_dates : list
                        future_dates : list
                        current_xreg : dict
                        future_xreg : dict
                        forecast_out_periods : int
        """
        self.name = name
        self.y = y
        self.current_dates = current_dates
        self.future_dates = future_dates
        self.current_xreg = current_xreg
        self.future_xreg = future_xreg
        self.forecast_out_periods = forecast_out_periods
        self.info = {}
        self.mape = {}
        self.forecasts = {}
        self.feature_importance = {}
        self.ordered_xreg = []
        self.best_model = ''

    def get_data_fred(self,series,date_start='1900-01-01'):
        """ imports data from FRED into a pandas dataframe
            stores the results in self.name, self.y, and self.current_dates
            Parameters: series : str
                            the name of the series to extract from FRED
                        date_start : str
                            the date to begin the time series
                            must be in YYYY-mm-dd format
        """
        self.name = series
        df = pdr.get_data_fred(series,start=date_start)
        self.y = list(df[series])
        self.current_dates = list(pd.to_datetime(df.index))
        
    def process_xreg_df(self,xreg_df,date_col=None,process_missing='remove'):
        """ take a dataframe of X regressors and store them in the appropriate self. locations
        """
        def _remove_(c):
            xreg_df.drop(columns=c,inplace=True)
        def _impute_mean_(c):
            xreg_df[c].fillna(xreg_df[c].mean(),inplace=True)
        def _impute_median_(c):
            xreg_df[c].fillna(xreg_df[c].median(),inplace=True)
        def _impute_mode_(c):
            from scipy.stats import mode
            xreg_df[c].fillna(mode(xreg_df[c])[0][0],inplace=True)
        def _forward_fill_(c):
            xreg_df[c].fillna(method='ffill',inplace=True)
        def _backward_fill_(c):
            xreg_df[c].fillna(method='bfill',inplace=True)
        def _impute_w_nearest_neighbors_(c):
            from sklearn.neighbors import KNeighborsClassifier
            predictors=[e for e in xreg_df if len(xreg_df[e].dropna())==len(xreg_df[e])] # predictor columns can have no NAs
            predictors=[e for e in predictors if e != c] # predictor columns cannot be the same as the column to impute (this should be taken care of in the line above, but jic)
            predictors=[e for e in predictors if xreg_df[e].dtype in (np.int32,np.int64,np.float32,np.float64,int,float)] # predictor columns must be numeric -- good idea to dummify as many columns as possible
            clf = KNeighborsClassifier(3, weights='distance')
            df_complete = xreg_df.loc[xreg_df[c].isnull()==False]
            df_nulls = xreg_df.loc[xreg_df[c].isnull()]
            trained_model = clf.fit(df_complete[predictors],df_complete[c])
            imputed_values = trained_model.predict(df_nulls[predictors])
            df_nulls[c] = imputed_values
            xreg_df[c] = pd.concat(df_complete[c],df_nulls[c])

        if not date_col is None:
            xreg_df[date_col] = pd.to_datetime(xreg_df[date_col])
            self.future_dates = list(xreg_df.loc[xreg_df[date_col] > self.current_dates[-1],date_col])
            xreg_df = xreg_df.loc[xreg_df[date_col] >= self.current_dates[0]]
        xreg_df = pd.get_dummies(xreg_df,drop_first=True)

        if isinstance(process_missing,dict):
            for c, v in process_missing.items():
                if (v == 'remove') & (xreg_df[c].isnull().sum() > 0):
                    _remove_(c)
                elif v == 'impute_mean':
                    _impute_mean_(c)
                elif v == 'impute_median':
                    _impute_median_(c)
                elif v == 'impute_mode':
                    _impute_mode_(c)
                elif v == 'forward_fill':
                    _forward_fill_(c)
                elif v == 'backward_fill':
                    _backward_fill_(c)
                elif v == 'impute_w_nearest_neighbors':
                    _impute_w_nearest_neighbors_(c)
                else:
                    print(f'argument {process_missing} not supported')
            remaining_c = [c for c in xreg_df if c not in process_missing.keys()]
            for c in remaining_c:
                if xreg_df[c].isnull().sum() > 0:
                    _remove_(c)

        elif isinstance(process_missing,str):
            for c in xreg_df:
                if xreg_df[c].isnull().sum() > 0:
                    if process_missing == 'remove':
                        _remove_(c)
                    elif prcoess_missing == 'impute_mean':
                        _impute_mean_(c)
                    elif prcoess_missing == 'impute_median':
                        _impute_median_(c)
                    elif prcoess_missing == 'impute_mode':
                        _impute_mode_(c)
                    elif prcoess_missing == 'forward_fill':
                        _forward_fill_(c)
                    elif prcoess_missing == 'backward_fill':
                        _backward_fill_(c)
                    elif prcoess_missing == 'impute_w_nearest_neighbors':
                        _impute_w_nearest_neighbors_(c)
                    else:
                        print(f'argument {process_missing} not supported')
                        return None

        if not date_col is None:
            current_xreg_df = xreg_df.loc[xreg_df[date_col].isin(self.current_dates)].drop(columns=date_col)
            future_xreg_df = xreg_df.loc[~xreg_df[date_col].isin(self.current_dates)].drop(columns=date_col)        
        else:
            current_xreg_df = xreg_df.iloc[:len(self.y)]
            future_xreg_df = xreg_df.iloc[len(self.y):]

        assert current_xreg_df.shape[0] == len(self.y)

        self.forecast_out_periods = future_xreg_df.shape[0]
        self.current_xreg = {}
        self.future_xreg = {}
        for c in current_xreg_df:
            self.current_xreg[c] = list(current_xreg_df[c])
            self.future_xreg[c] = list(future_xreg_df[c])
        
    def forecast_arima(self):
        pass

    def forecast_tbats(self):
        pass

    def forecast_rf(self,test_length=1,Xvars='all',call_me='rf',hyper_params={},set_feature_importance=True):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        X = pd.DataFrame(self.current_xreg)
        y = pd.Series(self.y)
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_length,shuffle=False)
        
        regr = RandomForestRegressor(**hyper_params,random_state=20)
        regr.fit(X_train,y_train)
        pred = regr.predict(X_test)
        
        test_set_ape = [np.abs(yhat-y) / y for yhat, y in zip(pred,y_test)]
        
        regr.fit(X,y)
        
        new_data = pd.DataFrame(self.future_xreg)
        
        f = regr.predict(new_data)
        
        self.info[call_me] = {}
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = 'Random Forest {}'.format(hyper_params)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = list(pred)
        self.info[call_me]['test_set_ape'] = test_set_ape
        
        self.mape[call_me] = np.array(test_set_ape).mean()
        self.forecasts[call_me] = list(f)