import pandas as pd
import numpy as np
import os
import pandas_datareader as pdr

rwd = os.getcwd().replace('\\','/')

class Data:
    def __init__(self,name=None,y=None,current_dates=None,future_dates=None,
                 current_xreg=None,future_xreg=None,forecast_out_periods=24):
        """ Parameters: name : str
                        y : list
                        current_dates : list
                            an ordered list of dates that correspond to the ordered values in self.y
                            dates must be a pandas datetime object (pd.to_datetime())
                        future_dates : list
                            dates must be a pandas datetime object (pd.to_datetime())
                        current_xreg : dict
                        future_xreg : dict
                        forecast_out_periods : int, default 24
        """
        self.name = name
        self.y = y
        self.current_dates = current_dates
        self.future_dates = future_dates
        self.current_xreg = current_xreg
        self.future_xreg = future_xreg
        self.forecast_out_periods=forecast_out_periods
        self.info = {}
        self.mape = {}
        self.forecasts = {}
        self.feature_importance = {}
        self.best_model = ''

    def _score_and_forecast(self,call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars):
        """ scores a model on a test test
            forecasts out however many periods in the new data (stored in self.new_yXdf)
            writes info to self.info, self.mape, and self.forecasts (this process is described in more detail in the forecast methods)
            only works within an sklearn forecast method
            Parameters: call_me : str
                        regr : sklearn.<regression_model>
                        X : pd.core.frame.DataFrame
                        y : pd.Series
                        X_train : pd.core.frame.DataFrame
                        y_train : pd.Series
                        X_test : pd.core.frame.DataFrame
                        y_test : pd.Series
                        Xvars : list or any other data type
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
        """ 
        regr.fit(X_train,y_train)
        pred = regr.predict(X_test)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = list(pred)
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / y for yhat, y in zip(pred,y_test)]
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()
        regr.fit(X,y)
        new_data = pd.DataFrame(self.future_xreg)
        if isinstance(Xvars,list):
            new_data = new_data[Xvars]
        f = regr.predict(new_data)
        self.forecasts[call_me] = list(f)

    def _set_remaining_info(self,call_me,test_length,model_name,hyper_params):
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = f"{model_name} - {hyper_params}"

    def _train_test_split(self,test_length,Xvars='all'):
        """ returns an X/y full set, training set, and testing set
            resulting y objects are pd.Series
            resulting X objects are pd.core.frame.DataFrame
            this is a non-random split and the resulting test set will be size specified in test_length
            only works within an sklearn forecast method
            Parameters: test_length : int,
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables to use in the resulting X dataframes
                            if it is not a list, it is assumed that all variables are wanted
        """
        from sklearn.model_selection import train_test_split
        X = pd.DataFrame(self.current_xreg)
        if isinstance(Xvars,list):
            X = X[Xvars]  
        y = pd.Series(self.y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_length,shuffle=False)
        return X, y, X_train, X_test, y_train, y_test

    def _set_feature_importance(self,X,y,regr):
        """ returns the permutation feature importances of any regression model in a pandas dataframe
            leveraged eli5 package (https://pypi.org/project/eli5/)
            only works within an sklearn forecast method
            Parameters: X : pd.core.frame.DataFrame
                            X regressors used to predict the depdendent variable
                        y : pd.Series
                            y series representing the known values of the dependent variable
                        regr : sklearn estimator
                            the estimator to use to score the permutation feature importance
        """
        import eli5
        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(regr).fit(X,y)
        weights_df = eli5.explain_weights_df(perm,feature_names=X.columns.tolist())
        return weights_df

    def get_data_fred(self,series,date_start='1900-01-01'):
        """ 
        """
        self.name = series
        df = pdr.get_data_fred(series,start=date_start)
        df.to_csv('tmp/tmp.csv')
        self.y = list(df[series])
        self.current_dates = list(pd.to_datetime(df.index))

    def process_xreg_df(self,xreg_df,date_col=None,process_missing='remove'):
        """ takes a dataframe of external regressors
            any non-numeric data will be made into a 0/1 binary variable
            deals with columns with missing data
            eliminates rows that don't correspond with self.y's timeframe
            splits values between the future and current xregs
            changes self.forecast_out_periods
            for more complex processing, perform manipulations before passing through this function
            Parameters: xreg_df : pandas dataframe
                        date_col : str, default None
                            the date column in xreg_df, if applicable
                            if None, assumes none of the columns are dates
                            if None, assumes all dataframe obs start at the same time as self.y
                            if str, will use to resize the dataset to line up with self.current_dates
                        process_missing : str or dict,
                                          if str, one of 'remove','impute_mean','impute_median','impute_mode','forward_fill','backward_fill','impute_w_nearest_neighbors'; default 'remove'
                                          if dict, key is a column name and value is one of 'remove','impute_mean','impute_median','impute_mode','forward_fill','backward_fill','impute_w_nearest_neighbors'
                                          if str, one method applied to all columns
                                          if dict, the selected methods only apply to column names in the dictionary
                                          warning when using dict, all columns with missing data not in dict will be removed
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
            self.future_dates = list(xreg_df.loc[xreg_df[date_col] > self.current_dates[-1],'Date'])
            xreg_df = xreg_df.loc[xreg_df[date_col] >= self.current_dates[0]]
        xreg_df = pd.get_dummies(xreg_df,drop_first=True)
        xreg_df = xreg_df

        if isinstance(process_missing,dict):
            for c, v in process_missing.items():
                if v == 'remove':
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

    def check_xreg_future_current_consistency(self):
        """ checks...
            if any of these checks fails, raises an error
        """
        for k, v in self.current_xreg.items():
            assert len(self.y) == len(self.current_dates)
            assert len(self.current_xreg[k]) == len(self.y)
            assert k in self.future_xreg.keys()
            assert len(self.future_xreg[k]) == len(self.future_dates)

    def set_forecast_out_periods(self,n):
        if isinstance(n,int):
            self.forecast_out_periods = n
        else:
            print('n must be an int type')

    def forecast_arima(self):
        pass

    def forecast_tbats(self):
        pass

    def forecast_rf(self,test_length=1,Xvars='all',call_me='rf',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a random forest from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_period' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Random Forest) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "rf"
                            what to call the evaluated model -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict or str, if str, must be "default"; default "default"
                            any hyper paramaters that you want changed from the default setting from sklearn, dictionary where parameter is key, desired setting is value
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.ensemble import RandomForestRegressor
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = RandomForestRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Level Perceptron',str(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_gbt(self):
        pass

    def forecast_adaboost(self):
        pass

    def forecast_mlp(self,test_length=1,Xvars='all',call_me='mlp',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a multi level perceptron neural network from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_period' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Multi Level Perceptron) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "mlp"
                            what to call the evaluated model -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, dictionary where parameter is key, desired setting is value
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.neural_network import MLPRegressor
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = MLPRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Level Perceptron',str(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlr(self):
        pass

    def forecast_ridge(self):
        pass

    def forecast_lasso(self):
        pass

    def forecast_svr(self):
        pass

    def forecast_average(self):
        pass

    def set_best_model(self):
        pass

    def get_model_order_best_to_worst(self):
        pass