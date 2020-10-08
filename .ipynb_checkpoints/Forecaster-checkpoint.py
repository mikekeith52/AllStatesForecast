import pandas as pd
import numpy as np
import os
import pandas_datareader as pdr
from collections import Counter
from scipy.stats import pearsonr
import rpy2.robjects as ro

rwd = os.getcwd().replace('\\','/')

class Forecaster:
    """ object to forecast time series data
        natively supports the extraction of FRED data, could be expanded to finance data with few adjustments
        The following models are currently supported:
            random forest (sklearn)
            adaboost (sklearn)
            gradient boosted trees (sklearn)
            support vector regressor (sklearn)
            ridge (sklearn)
            lasso (sklearn)
            multi linear regression (sklearn)
            multi level perceptron (sklearn)
            arima (R forecast pkg: auto.arima with or without external regressors)
            tbats (R forecast pkg: tbats)
            average (any number of models can be averaged)
        Author Michael Keith: mikekeith52@gmail.com
    """
    def __init__(self,name=None,y=None,current_dates=None,future_dates=None,
                 current_xreg=None,future_xreg=None,forecast_out_periods=24):
        """ Parameters: name : str
                        y : list
                        current_dates : list
                            an ordered list of dates that correspond to the ordered values in self.y
                            dates must be a pandas datetime object (pd.to_datetime())
                        future_dates : list
                            an ordered list of dates that correspond to the future periods being forecasted
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
        self.ordered_xreg = None
        self.forecast_out_periods=forecast_out_periods
        self.info = {}
        self.mape = {}
        self.forecasts = {}
        self.feature_importance = {}
        self.ordered_xreg = []
        self.best_model = ''

    def _score_and_forecast(self,call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars):
        """ scores a model on a test test
            forecasts out however many periods in the new data
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

    def _set_remaining_info(self,call_me,test_length,model_form):
        """ sets the holdout_periods and model_form values in the self.info nested dictionary, where call_me (model nickname) is the key
        """
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = model_form

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
            leverages eli5 package (https://pypi.org/project/eli5/)
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

    def _prepr(self,*args,write_Xvars='all'):
        """ prepares the R environment by importing/installing libraries and writing out csv files (current/future datasets) to tmp folder
            file names: tmp_r_current.csv, tmp_r_future.csv
            args are libs to import and/or install from R
            Parameters: args : str
                            library names to import into the R environment
                            if library name not found, will attempt to install (you will need to specify a CRAN mirror in a pop-up box)
                        write_Xvars : str or list, if str must be "all", default "all"
                            Xvars to write to the tmp folder in the csv files
        """
        from rpy2.robjects.packages import importr
        for lib in args:
            try:  importr(lib)
            except: ro.r(f'install.packages("{lib}")') ; importr(lib)
        current_df = pd.DataFrame(self.current_xreg)
        current_df['y'] = self.y

        if isinstance(write_Xvars,list):
            current_df = current_df[['y'] + write_Xvars]
        elif write_Xvars is None:
            current_df = current_df['y']
        elif write_Xvars != 'all':
            print(f'unknown argument passed to write_Xvars: {write_Xvars}')
            return None

        if 'tmp' not in os.listdir():
            os.mkdir('tmp')

        current_df.to_csv(f'tmp/tmp_r_current.csv',index=False)
        
        if not write_Xvars is None:
            future_df = pd.DataFrame(self.future_xreg)
            future_df.to_csv(f'tmp/tmp_r_future.csv',index=False)

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
        """ takes a dataframe of external regressors
            any non-numeric data will be made into a 0/1 binary variable (using pd.get_dummies())
            deals with columns with missing data
            eliminates rows that don't correspond with self.y's timeframe
            splits values between the future and current xregs
            changes self.forecast_out_periods
            assumes the dataframe is aggregated to the same timeframe as self.y (monthly, quarterly, etc.)
            for more complex processing, perform manipulations before passing through this function
            stores results in self.xreg
            Parameters: xreg_df : pandas dataframe
                        date_col : str, default None
                            the date column in xreg_df, if applicable
                            if None, assumes none of the columns are dates (if this is not the case, can cause bad results)
                            if None, assumes all dataframe obs start at the same time as self.y
                            if str, will use that column to resize the dataset to line up with self.current_dates and all future obs stored in self.future_xreg and self.future_dates
                        process_missing : str or dict,
                                          if str, one of 'remove','impute_mean','impute_median','impute_mode','forward_fill','backward_fill','impute_w_nearest_neighbors'; default 'remove'
                                          if dict, key is a column name and value is one of 'remove','impute_mean','impute_median','impute_mode','forward_fill','backward_fill','impute_w_nearest_neighbors'
                                          if str, one method applied to all columns
                                          if dict, the selected methods only apply to column names in the dictionary
                                          note when using dict, all columns with missing data not in dict will be removed
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

    def set_and_check_data_types(self,check_xreg=True):
        """ changes all attributes in self to lists, dicts, strs, or whatever makes it easier for the program to work with with no errors
            if a conversion to these types is unsuccessfully, will raise an error
            Parameters: check_xreg : bool, default True
                if True, checks that self.current_xreg and self.future_xregs are dict types
                if not, raises an error
                change this to false if wanting to perform auto-regressive forecasts only
        """
        self.name = str(self.name) if not isinstance(self.name,str) else self.name
        self.y = list(self.y) if not isinstance(self.y,list) else self.y
        self.current_dates = list(self.current_dates) if not isinstance(self.current_dates,list) else self.current_dates
        self.future_dates = list(self.future_dates) if not isinstance(self.future_dates,list) else self.future_dates
        self.forecast_out_periods = int(self.forecast_out_periods)
        if check_xreg:
            assert isinstance(self.current_xreg,dict)
            assert isinstance(self.future_xreg,dict)
        
    def check_xreg_future_current_consistency(self):
        """ checks that the self.y is same size as self.current_dates
            checks that self.y is same size as the values in self.current_xreg
            checks that self.future_dates is same size as the values in self.future_xreg
            if any of these checks fails, raises an AssertionError
        """
        for k, v in self.current_xreg.items():
            assert len(self.y) == len(self.current_dates)
            assert len(self.current_xreg[k]) == len(self.y)
            assert k in self.future_xreg.keys()
            assert len(self.future_xreg[k]) == len(self.future_dates)

    def set_forecast_out_periods(self,n):
        """ sets the self.forecast_out_periods attribute and changes self.future_dates and self.future_xreg appropriately
            Parameters: n : int
                the number of periods you want to forecast out for
                if this is a larger value than the size of self.future_dates, some models may fail
        """
        if isinstance(n,int):
            self.forecast_out_periods = n
            if isinstance(self.future_dates,list):
                self.future_dates = self.future_dates[:n]
            if isinstance(self.future_xreg,dict):
                for k,v in self.future_xreg.items():
                    self.future_xreg[k] = v[:n]
        else:
            print('n must be an int type')

    def set_ordered_xreg(self,chop_tail_periods=0,include_only='all',exclude=None,quiet=True):
        """ method for ordering stored externals from most to least correlated, according to absolute Pearson correlation coefficient value
            will not error out if a given external has no variation in it -- will simply skip
            when measuring correlation, will log/difference variables when possible to compare stationary results
            stores the results in self.ordered_xreg as a list
            Parameters: chop_tail_periods : int, default 0
                            The number of periods to chop from the respective membermonths and externals list
                            This is used to reduce the chance of overfitting the data by using mismatched test periods for forecasts
                        include_only : list or any other data type, default "all"
                            if this is a list, only the externals in the list will be considered when testing correlation
                            if this is not a list, then it will be ignored and all externals will be considered
                            if this is a list, exclude will be ignored
                        exclude : list or any other data type, default None
                            if this is a list, the externals in the list will be excluded when testing correlation
                            if this is not a ist, then it will be ignored and no extenrals will be excluded
                            if include_only is a list, this is ignored
                            note: is possible for include_only to be its default value, "all", and exclude to not be ignored if it is passed as a list type
                        quiet : bool or any other data type, default True
                            if this is True, then if a given external is ignored (either because no correlation could be calculated or there are no observations after its tail has been chopped), you will not know
                            if this is not Ture, then if a given external is ignored, it will print which external is being skipped
        """
        def log_diff(x):
            """ returns the logged difference of an array
            """
            return np.diff(np.log(x),n=1)

        if isinstance(include_only,list):
            use_these_externals = {}
            for e in include_only:
                use_these_externals[e] = self.current_xreg[e]
        else:
            use_these_externals = self.current_xreg
            if isinstance(exclude,list):
                for e in exclude:
                    use_these_externals.pop(e)
                
        ext_reg = {}
        for k, v in use_these_externals.items():
            if chop_tail_periods > 0:
                x = np.array(v[:(chop_tail_periods*-1)])
                y = np.array(self.y[:(chop_tail_periods*-1)])
            else:
                x = np.array(v)
                y = np.array(self.y)
                
            if (x.min() <= 0) & (y.min() > 0):
                y = log_diff(y)
                x = x[1:]
            elif (x.min() > 0) & (y.min() > 0):
                y = log_diff(y)
                x = log_diff(x)
            
            if len(np.unique(x)) == 1:
                if not quiet:
                    print(f'no variation in {k} for time period specified')
                continue
            else: 
                r_coeff = pearsonr(y,x)
            
            if np.abs(r_coeff[0]) not in ext_reg.values():
                ext_reg[k] = np.abs(r_coeff[0])
        
        k = Counter(ext_reg) 
        self.ordered_xreg = [h[0] for h in k.most_common()] # this should give us the ranked external regressors

    def forecast_arima(self,test_length=1,Xvars='all',call_me='arima'):
        """ forecasts using auto.arima from the forecast package in R
            evaluates the period MAPE with MLmetrics
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_period' : int - the number of periods held out in the test set
                        'model_form' : str - the final evaluated arima form to create the forecast (defined as the forecast()[[1]] element from the forecast package)
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "arima"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
        """
        if isinstance(Xvars,str):
            if Xvars.startswith('top_'):
                top_xreg = int(Xvars.split('_')[1])
                if top_xreg == 0:
                    Xvars = None
                else:
                    self.set_ordered_xreg(chop_tail_periods=test_length) # have to reset here for differing test lengths in other models
                    Xvars = self.ordered_xreg[:top_xreg]
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('MLmetrics','forecast',write_Xvars=Xvars)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')
            data_train <- data[1:(nrow(data)-{test_length}),]
            data_test <- data[(nrow(data)-{test_length} + 1):nrow(data),]
            
            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
            
            """)
        ro.r("""
            if (ncol(data) > 1){
                future_externals = read.csv('tmp/tmp_r_future.csv')
                externals = names(data)[2:ncol(data)]
                xreg_c <- as.matrix(data[,externals])
                xreg_tr <- as.matrix(data_train[,externals])
                xreg_te <- as.matrix(data_test[,externals])
                xreg_f <- as.matrix(future_externals[,externals])
            } else {
                xreg_c <- NULL
                xreg_tr <- NULL
                xreg_te <- NULL
                xreg_f <- NULL
            }
            ar <- auto.arima(y_train,xreg=xreg_tr)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[4]] are point estimates, f[[1]] is the ARIMA form
            p <- f[[4]]
            arima_form <- f[[1]]
            mape <- MLmetrics::MAPE(p,y_test)
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / write$actual
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)
        """)
        
        ro.r(f"""
            ar <- auto.arima(y,max.order=10,stepwise=F,xreg=xreg_c)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[4]]
            arima_form <- f[[1]]
            
            write <- data.frame(forecast=p)
            write$expected_mape <- mape
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])

    def forecast_tbats(self,test_length=1,Xvars='all',call_me='tbats'):
        """ forecasts using tbats from the forecast package in R
            evaluates the period MAPE with MLmetrics
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_period' : int - the number of periods held out in the test set
                        'model_form' : str - the final evaluated arima form to create the forecast (defined as the forecast()[[1]] element from the forecast package)
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1
                        call_me : str, default "tbats"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
        """
        if isinstance(Xvars,str):
            if Xvars.startswith('top_'):
                top_xreg = int(Xvars.split('_')[1])
                if top_xreg == 0:
                    Xvars = None
                else:
                    self.set_ordered_xreg(chop_tail_periods=test_length) # have to reset here for differing test lengths in other models
                    Xvars = self.ordered_xreg[:top_xreg]
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('MLmetrics','forecast',write_Xvars=None)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')

            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
        
            ar <- tbats(y_train)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[2]] are point estimates, f[[9]] is the TBATS form
            p <- f[[2]]
            tbats_form <- f[[9]]
            mape <- MLmetrics::MAPE(p,y_test)
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / write$actual
            write$model_form <- tbats_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- tbats(y)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[2]]
            tbats_form <- f[[9]]
            
            write <- data.frame(forecast=p)
            write$expected_mape <- mape
            write$model_form <- tbats_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])

    def forecast_ets(self,test_length=1,Xvars='all',call_me='ets'):
        """ forecasts using ets from the forecast package in R
            evaluates the period MAPE with MLmetrics
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_period' : int - the number of periods held out in the test set
                        'model_form' : str - the final evaluated arima form to create the forecast (defined as the forecast()[[1]] element from the forecast package)
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1
                        call_me : str, default "ets"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
        """
        if isinstance(Xvars,str):
            if Xvars.startswith('top_'):
                top_xreg = int(Xvars.split('_')[1])
                if top_xreg == 0:
                    Xvars = None
                else:
                    self.set_ordered_xreg(chop_tail_periods=test_length) # have to reset here for differing test lengths in other models
                    Xvars = self.ordered_xreg[:top_xreg]
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('MLmetrics','forecast',write_Xvars=None)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')

            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
        
            ar <- ets(y_train)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[2]] are point estimates, f[[9]] is the TBATS form
            p <- f[[2]]
            ets_form <- f[[8]]
            mape <- MLmetrics::MAPE(p,y_test)
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / write$actual
            write$model_form <- ets_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- ets(y)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[2]]
            ets_form <- f[[8]]
            
            write <- data.frame(forecast=p)
            write$expected_mape <- mape
            write$model_form <- ets_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])

    def forecast_rf(self,test_length=1,Xvars='all',call_me='rf',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a random forest from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Random Forest) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict; default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argment collection (**hyper_params) to the sklearn model
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
        self._set_remaining_info(call_me,test_length,'Random Forest - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_gbt(self,test_length=1,Xvars='all',call_me='gbt',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a gradient boosting regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Random Forest) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict; default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argment collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.ensemble import GradientBoostingRegressor
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = GradientBoostingRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Gradient Boosted Trees - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_adaboost(self,test_length=1,Xvars='all',call_me='adaboost',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with an ada boost regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Random Forest) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict; default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argment collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.ensemble import AdaBoostRegressor
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = AdaBoostRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Ada Boosted Trees - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlp(self,test_length=1,Xvars='all',call_me='mlp',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a multi level perceptron neural network from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Multi Level Perceptron) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "mlp"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict; default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argment collection (**hyper_params) to the sklearn model
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
        self._set_remaining_info(call_me,test_length,'Multi Level Perceptron - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlr(self,test_length=1,Xvars='all',call_me='mlr',set_feature_importance=True):
        """ forecasts the stored y variable with a multi linear regression from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - Multi Linear Regression
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "mlr"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.linear_model import LinearRegression
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = LinearRegression()
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Linear Regression')
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_ridge(self,test_length=1,Xvars='all',call_me='ridge',alpha=1.0,set_feature_importance=True):
        """ forecasts the stored y variable with a ridge regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Ridge) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "ridge"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        alpha : float, default 1.0
                            the desired alpha hyperparameter to pass to the sklearn model
                            1.0 is also the default in sklearn
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.linear_model import Ridge
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = Ridge(alpha=alpha)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Ridge - {}'.format(alpha))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_lasso(self,test_length=1,Xvars='all',call_me='lasso',alpha=1.0,set_feature_importance=True):
        """ forecasts the stored y variable with a lasso regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Lasso) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "lasso"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        alpha : float, default 1.0
                            the desired alpha hyperparameter to pass to the sklearn model
                            1.0 is also the default in sklearn
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.linear_model import Lasso
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = Lasso(alpha=alpha)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Lasso - {}'.format(alpha))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_svr(self,test_length=1,Xvars='all',call_me='svr',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a support vector regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model (Support Vector Regressor) and any user-specified hyperparameters if different from the default setting from sklearn
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
                in self.forecasts, a key is added as the model name (specified in call_me), and a list of forecasted figures as the value
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                        Xvars : list or any other data type, default "all"
                            the independent variables used to make predictions
                            if it is not a list, it is assumed that all variables are wanted
                        call_me : str, default "mlp"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict; default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argment collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importancere with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
        """
        from sklearn.svm import SVR
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = SVR(**hyper_params)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Support Vector Regressor - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)


    def forecast_average(self,models='all',call_me='average',test_length='max'):
        """ averages a set of models to make a new estimator
            the following information is stored:
                in self.info, a key is added as the model name (specified in call_me), and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - this will always take the form of:
                                             "Average of" + f"{number_of_models}" + "models :" + f"{name_of_each_model_separated_by_spaces}"
                        'test_set_actuals' : list - the actual y figures from the test set
                        'test_set_predictions' : list - the predicted y figures forecasted from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape, a key is added as the model name (specified in call_me), and the MAPE as the value
                    MAPE defined as the mean APE (Absolute Percent Error) stored in self.info for the given model
            Parameters: models : str or list, if str one of "all" or starts with "top_", default "all"
                            "all" will average all models, except the naive model (named in the naive_is_called parameter) or any None models (models that errored out before being evaluated)
                            starts with "top_" will average the top however many models are specified according to their respective MAPE values on the test set (lower = better)
                                the character after "top_" must be an integer
                                ex. "top_5"
                            if list, then those are the models that will be averaged
                        call_me : str, default "average"
                            what to call the evaluated model -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        test_length : int or str, if str, must be "max"; default "max"
                            the test length to assign to the average model
                            if max, it will use the maximum test_length from all saved models
                            if int, will use that many test periods
                            if int is greater than one of the stored models' test length, this will fail
        """
        if models == 'all':
            avg_these_models = [e for e in list(self.mape.keys()) if (e != call_me) & (not e is None)]
        elif isinstance(models,list):
            avg_these_models = models.copy()
        elif isinstance(models,str):
            if models.startswith('top_'):
                ordered_models = [e for e in self.order_all_forecasts_best_to_worst() if (e != call_me) & (not e is None)]
                avg_these_models = []
                for i, m in enumerate(ordered_models):
                    if (i+1) <= int(models.split('_')[1]):
                        avg_these_models.append(m)
        else:
            print(f'argument in models parameter not recognized: {models}')
            return None
            
        if len(avg_these_models) == 0:
            print('no models found to average')
            return None

        if test_length == 'max':
            for i, m in enumerate(avg_these_models):
                if i == 0:
                    test_length = self.info[m]['holdout_periods']
                else:
                    test_length = min(test_length,self.info[m]['holdout_periods'])
        
        self.mape[call_me] = 1
        self.forecasts[call_me] = [None]*self.forecast_out_periods

        self.info[call_me] = {'holdout_periods':test_length,
                             'model_form':None,
                             'test_set_actuals':self.y[-(test_length):],
                             'test_set_predictions':[None],
                             'test_set_ape':[None]}

        model_forms = []
        forecasts = pd.DataFrame()
        test_set_predictions_df = pd.DataFrame()
        test_set_ape_df = pd.DataFrame()

        for m in avg_these_models:
            test_set_predictions_df[m] = self.info[m]['test_set_predictions'][-(test_length):]
            test_set_ape_df[m] = self.info[m]['test_set_ape'][-(test_length):] 
            model_forms.append(self.info[m]['model_form'])
            forecasts[m] = self.forecasts[m]
            
        self.info[call_me]['model_form'] = 'Average of ' + str(len(avg_these_models)) + ' models: ' + ' '.join(model_forms)
        self.info[call_me]['test_set_predictions'] = list(test_set_predictions_df.mean(axis=1))
        self.info[call_me]['test_set_ape'] = list(test_set_ape_df.mean(axis=1))
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()
        self.forecasts[call_me] = list(forecasts.mean(axis=1))

    def set_best_model(self):
        """ sets the best forecast model based on which model has the lowest MAPE value for the given holdout periods
            if two models tie, it will select 1 at random
        """
        self.best_model = Counter(self.mape).most_common()[-1][0]

    def order_all_forecasts_best_to_worst(self):
        """ returns a list of the evaluated models for the given series in order of best-to-worst according to their evaluated MAPE values
            using different-sized test sets for different models could cause some trouble here, but I don't see a better way
        """
        x = [h[0] for h in Counter(self.mape).most_common()]
        x.reverse()
        return x.copy()