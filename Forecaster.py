import pandas as pd
import numpy as np
import datetime
import os
import pandas_datareader as pdr
from collections import Counter
from scipy import stats
import rpy2.robjects as ro
from sklearn.metrics import r2_score

# make the working directory friendly for R
rwd = os.getcwd().replace('\\','/')

# add a descriptive error
class ForecastFormatError(Exception):
    pass

class Forecaster:
    """ object to forecast time series data
        natively supports the extraction of FRED data, could be expanded to other APIs with few adjustments
        the following models are supported:
            adaboost (sklearn)
            auto_arima (R forecast::auto.arima, no seasonal models)
            auto_arima_seas (R forecast::auto.arima, seasonal models)
            arima (statsmodels, not automatically optimized)
            sarimax13 (Seasonal Auto Regressive Integrated Moving Average by X13 - R seasonal::seas)
            average (any number of models can be averaged)
            ets (exponental smoothing state space model - R forecast::ets)
            gbt (gradient boosted trees - sklearn)
            hwes (holt-winters exponential smoothing - statsmodels hwes)
            auto_hwes (holt-winters exponential smoothing - statsmodels hwes)
            lasso (sklearn)
            mlp (multi level perceptron - sklearn)
            mlr (multi linear regression - sklearn)
            prophet (facebook prophet - fbprophet)
            rf (random forest - sklearn)
            ridge (sklearn)
            svr (support vector regressor - sklearn)
            tbats (exponential smoothing state space model With box-cox transformation, arma errors, trend, and seasonal component - R forecast::tbats)
            nnetar (time series neural network - R forecast::nnetar)
            var (vector auto regression - R vars::VAR)
            vecm (vector error correction model - R tsDyn::VECM)
        more models can be added by building more methods

        for every evaluated model, the following information is stored in the object attributes:
            in self.info (dict), a key is added as the model name and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model with any hyperparameters, external regressors, etc
                        'test_set_actuals' : list - the actual figures from the test set
                        'test_set_predictions' : list - the predicted figures from the test set evaluated with a model from the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                        'fitted_values' : list - the model's fitted values, when available. if not available, None
                in self.mape (dict), a key is added as the model name and the Mean Absolute Percent Error as the value
                in self.rmse (dict), a key is added as the model name and the Root Mean Square Error as the value
                in self.mae (dict), a key is added as the model name and the Mean Absolute Error as the value
                in self.r2 (dict), a key is added as the model name and the R Squared as the value
                in self.forecasts (dict), a key is added as the model name and a list of forecasted figures as the value
                in self.feature_importance (dict), a key is added to the dictionary as the model name and the value is a dataframe that gives some info about the features' prediction power
                    if it is an sklearn model, it will be permutation feature importance from the eli5 package (https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)
                    any other model, it is a dataframe with at least the names of the variables in the index, with as much summary statistical info as possible
                    if the model doesn't use external regressors, no key is added here

        Author Michael Keith: mikekeith52@gmail.com
    """
    def __init__(self,name=None,y=None,current_dates=None,future_dates=None,current_xreg=None,future_xreg=None,forecast_out_periods=24,**kwargs):
        """ You can load the object with data using __init__ or you can leave all default arguments and load the data with an attached API method (such as get_data_fred())
            Keep the object types consistent (do not use tuples or numpy arrays instead of lists, for example)
            Parameters: name : str
                        y : list
                        current_dates : list
                            an ordered list of dates that correspond to the ordered values in self.y
                            elements must be able to be parsed by pandas as dates
                        future_dates : list
                            an ordered list of dates that correspond to the future periods being forecasted
                            elements must be able to be parsed by pandas as dates
                        current_xreg : dict
                        future_xreg : dict
                        forecast_out_periods : int, default length of future_dates or 24 if that is None
                        all keyword arguments become attributes
        """
        self.name = name
        self.y = y
        self.current_dates = current_dates
        self.future_dates = future_dates
        self.current_xreg = current_xreg
        self.future_xreg = future_xreg
        self.forecast_out_periods = forecast_out_periods if future_dates is None else len(future_dates)
        self.info = {}
        self.mape = {}
        self.rmse = {}
        self.mae = {}
        self.r2 = {}
        self.forecasts = {}
        self.feature_importance = {}
        self.ordered_xreg = None
        self.best_model = ''

        for key, value in kwargs.items():
            setattr(self,key,value)

    def _score_and_forecast(self,call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars):
        """ scores a model on a test test (sklearn models specific)
            forecasts out however many periods in the new data
            writes info to self.info, self.mape, and self.forecasts (this process is described in more detail in the forecast methods)
            Parameters: call_me : str
                        regr : sklearn.<regression_model>
                        X : pd.core.frame.DataFrame
                        y : pd.Series
                        X_train : pd.core.frame.DataFrame
                        y_train : pd.Series
                        X_test : pd.core.frame.DataFrame
                        y_test : pd.Series
                        Xvars : list or str
        """
        regr.fit(X_train,y_train)
        pred = regr.predict(X_test)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = list(pred)
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(pred,y_test)]
        self._metrics(call_me)
        regr.fit(X,y)
        new_data = pd.DataFrame(self.future_xreg)
        if isinstance(Xvars,list):
            new_data = new_data[Xvars]
        f = regr.predict(new_data)
        self.forecasts[call_me] = list(f)
        self.info[call_me]['fitted_values'] = list(regr.predict(X))

    def _set_remaining_info(self,call_me,test_length,model_form):
        """ sets the holdout_periods and model_form values in the self.info nested dictionary, where call_me (model nickname) is the key
        """
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = model_form

    def _train_test_split(self,test_length,Xvars='all'):
        """ returns an X/y full set, training set, and testing set (sklearn models specific)
            resulting y objects are pandas series
            resulting X objects are pandas dataframe
            this is a non-random split and the resulting test set will be size specified in test_length
            Parameters: test_length : int,
                            the length of the resulting test_set
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
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
            sklearn models specific
            Parameters: X : pandas dataframe
                            X regressors used to predict the depdendent variable
                        y : pandas series
                            y series representing the known values of the dependent variable
                        regr : sklearn estimator
                            the estimator to use to score the permutation feature importance
        """
        import eli5
        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(regr).fit(X,y)
        weights_df = eli5.explain_weights_df(perm,feature_names=X.columns.tolist()).set_index('feature')
        return weights_df

    def _prepr(self,*libs,test_length,call_me,Xvars):
        """ prepares the R environment by importing/installing libraries and writing out csv files (current/future datasets) to tmp folder
            file names: tmp_r_current.csv, tmp_r_future.csv
            *libs are libs to import and/or install from R
            Parameters: call_me : str
                        Xvars : list, "all", or starts with "top_"
                        *libs: str
                            library names to import into the R environment
                            if library name not found in R environ, will attempt to install it (you will need to specify a CRAN mirror in a pop-up box)
        """
        from rpy2.robjects.packages import importr
        if isinstance(Xvars,str):
            if Xvars.startswith('top_'):
                top_xreg = int(Xvars.split('_')[1])
                if top_xreg == 0:
                    Xvars = None
                else:
                    self.set_ordered_xreg(chop_tail_periods=test_length) # have to reset here for differing test lengths in other models
                    if top_xreg > len(self.ordered_xreg):
                        Xvars = self.ordered_xreg[:]
                    else:
                        Xvars = self.ordered_xreg[:top_xreg]
            elif Xvars == 'all':
                Xvars = list(self.current_xreg.keys())
            else:
                print(f'Xvars argument not recognized: {Xvars}, changing to None')
                Xvars = None

        for lib in libs:
            try:  importr(lib)
            except: ro.r(f'install.packages("{lib}")') ; importr(lib) # install then import the r lib
        current_df = pd.DataFrame(self.current_xreg)
        current_df['y'] = self.y

        if isinstance(Xvars,list):
            current_df = current_df[['y'] + Xvars] # reorder columns 
        elif Xvars is None:
            current_df = current_df['y']
        elif Xvars != 'all':
           raise ValueError(f'unknown argument passed to Xvars: {Xvars}')

        if 'tmp' not in os.listdir():
            os.mkdir('tmp')

        current_df.to_csv(f'tmp/tmp_r_current.csv',index=False)
        
        if not Xvars is None:
            future_df = pd.DataFrame(self.future_xreg)
            future_df.to_csv(f'tmp/tmp_r_future.csv',index=False)

    def _sm_summary_to_fi(self,sm_model,call_me):
        """ places summary output from a stasmodels model into self.feature_importance as a pandas dataframe
            https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe/52976810
        """
        results_summary = sm_model.summary()
        results_as_html = results_summary.tables[1].as_html()
        self.feature_importance[call_me] = pd.read_html(results_as_html, header=0, index_col=0)[0]

    def _get_info_dict(self):
        return dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape','fitted_values'])

    def _metrics(self,call_me):
        """ creates mape, rmse, mae, and r2
        """
        self.mape[call_me] = np.mean(self.info[call_me]['test_set_ape'])
        self.rmse[call_me] = np.mean([(y - yhat)**2 for yhat,y in zip(self.info[call_me]['test_set_predictions'],self.info[call_me]['test_set_actuals'])])**0.5
        self.mae[call_me] = np.mean([np.abs(y - yhat) for yhat,y in zip(self.info[call_me]['test_set_predictions'],self.info[call_me]['test_set_actuals'])])
        # r2 needs at least 2 observations to work, so test_length = 1 will not evaluate
        if len(self.info[call_me]['test_set_predictions']) > 1:
            self.r2[call_me] = r2_score(self.info[call_me]['test_set_predictions'],self.info[call_me]['test_set_actuals'])
        else:
            self.r2[call_me] = None

    def _ready_for_forecast(self,need_xreg=False):
        """ runs before each attempted to forecast to make sure:
                y is set as a list of numeric figures
                current_dates is set as a list of datetime objects
                future_dates is set as a list of datetime objects
                if current_xreg is set, future_xreg is also set and both are dictionaries with lists as values
        """  
        _no_error_ = 'before forecasting, the following issues need to be corrected:'
        error = _no_error_
        if isinstance(self.y,list):
            try:
                [float(i) for i in self.y]
            except ValueError:
                error+='\n  all elements in y attribute must be numeric'
        else:
            error+=f'\n  y attribute must be a list, not {type(self.y)}'

        dates = {'current_dates':self.current_dates,'future_dates':self.future_dates}
        for k,v in dates.items():
            if isinstance(v,list):
                try:
                    if k == 'current_dates':
                        self.current_dates = pd.to_datetime(v).to_list()
                    else:
                        self.future_dates = pd.to_datetime(v).to_list()
                except:
                    error+=f'\n  the elements in the {k} attribute must be able to be parsed by pandas date parser -- try passing each element in yyyy-mm-dd format'
            else:
                error+=f'\n  {k} attribute must be a list, got {type(v)}'

        if not isinstance(self.forecast_out_periods,int):
            error+=f'\n  the forecast_out_periods attribute must be an integer type, found {type(self.forecast_out_periods)}'
        else:
            if self.forecast_out_periods < 1:
                error+=f'\n  forecast_out_periods must be at least 1 (is {self.forecast_out_periods})'

        if (self.current_xreg is None) & (need_xreg):
            error+='\n  the forecast you are attempting to run needs at least one external regressor to work, could not find any in the current_xreg attribute'
        else:
            if isinstance(self.current_xreg,dict):
                if not isinstance(self.future_xreg,dict):
                    error+=f'\n  if you are passing external regressors to the current_xreg attribute, the future_xreg attribute must also be a dictionary type, found {type(self.future_xreg)} type'
                else:
                    for k in self.current_xreg.keys():
                        if k not in self.future_xreg.keys():
                            error+=f'\n  all keys in the current_xreg attribute must also be present in the future_xreg attribute, could not find {k}'
                    for k, v in self.current_xreg.items():
                        if (not isinstance(v,list)) | (not isinstance(self.future_xreg[k],list)):
                            error+=f'\n  all values in the current_xreg and future_xreg dictionaries must be list types, check the {k} key'
                        elif len(v) != len(self.y):
                            error+=f'\n all values in the current_xreg dictionary must be the same length as the y attribute, check {k}'
                        elif len(self.future_xreg[k]) != self.forecast_out_periods:
                            error+=f'\n  all values in the future_xreg dictionary must be the same length as the forecast_out_periods attribute value ({self.forecast_out_periods}), check {k}'
                        elif np.isnan(v).sum() > 0:
                            error+=f'\n  cannot have missing values in any of the current_xreg values, check {k}'
                        elif np.isnan(self.future_xreg[k]).sum() > 0:
                            error+=f'\n  cannot have missing values in any of the future_xreg values, check {k}'
            elif not self.current_xreg is None:
                error+=f'\n  current_xreg attribute must be a dict type if attempting to use external regressors, or None if not, found {type(self.current_xreg)}'

        if len(error) > len(_no_error_):
            raise ForecastFormatError(error)

    def get_data_fred(self,series,i=0,date_start='1900-01-01'):
        """ imports data from FRED into a pandas dataframe
            stores the results in self.name, self.y, and self.current_dates
            Parameters: series : str
                            the name of the series to extract from FRED
                        i : int
                            the number of differences to take in the series to make it stationary
                        date_start : str
                            the date to begin the time series
                            must be in YYYY-mm-dd format
            f = Forecaster()
            f.get_data_fred('UTUR')
            print(f.y)
            >>> [5.8, 5.8, ..., 5.0, 4.1]

            print(f.current_dates)
            >>> [Timestamp('1976-01-01 00:00:00'), Timestamp('1976-02-01 00:00:00'), ..., Timestamp('2020-09-01 00:00:00'), Timestamp('2020-10-01 00:00:00')]
        """
        self.name = series
        df = pdr.get_data_fred(series,start=date_start)
        if i > 0:
            df[series] = df[series].diff(i)
        df.dropna(inplace=True)
        self.y = df[series].to_list()
        self.current_dates = df.index.to_list()

    def process_xreg_df(self,xreg_df,date_col,process_columns=False):
        """ takes a dataframe of external regressors
            any non-numeric data will be made into a 0/1 binary variable (using pandas.get_dummies(drop_first=True))
            deals with columns with missing data
            eliminates rows that don't correspond with self.y's timeframe
            splits values between future and current observations
            changes self.forecast_out_periods based on how many periods included in the dataframe past current_dates attribute
            assumes the dataframe is aggregated to the same timeframe as self.y (monthly, quarterly, etc.)
            for more complex processing, perform manipulations before passing through this function
            stores results in self.current_xreg, self.future_xreg, self.future_dates, and self.forecast_out_periods
            Parameters: xreg_df : pandas dataframe, required
                            this should include only independent regressors either in numeric form or that can be dummied into a 1/0 variable as well as a date column that can be parsed by pandas
                            do not include the dependent variable value
                        date_col : str, requried
                            the name of the date column in xreg_df that can be parsed with the pandas.to_datetime() function
                        process_columns : str, dict, or False; optional
                            how to process columns with missing data - most forecasts will not run when missing data is present in either xreg dict
                            supported: {'remove','impute_mean','impute_median','impute_mode','impute_min','impute_max',impute_0','forward_fill','backward_fill','impute_random'}
                            if str, must be one of supported and that method is applied to all columns with missing data
                            if dict, key is a column and value is one of supported, method only applied to columns with missing data                  
                            'impute_random' will fill in missing values with random draws from the same column
           
            xreg_df = pd.DataFrame({'date':['2020-01-01','2020-02-01','2020-03-01','2020-04-01']},'x1':[1,2,3,5],'x2':[1,3,3,3])
            f = Forecaster(y=[4,5,9],current_dates=['2020-01-01','2020-02-01','2020-03-01'])
            f.process_xreg_df(xreg_df,date_col='date')
            print(f.current_xreg)
            >>> {'x1':[1,2,3],'x2':[1,3,3]}

            print(f.future_xreg)
            >>> {'x1':[5],'x2':[3]}

            print(f.future_dates)
            >>> [Timestamp('2020-04-01 00:00:00')]

            print(f.forecast_out_periods)
            >>> 1
        """
        # for other processing methods, add a function here that follows the same pattern and it should flow down automatically
        def _remove_(c): xreg_df.drop(columns=c,inplace=True)
        def _impute_mean_(c): xreg_df[c].fillna(xreg_df[c].mean(),inplace=True)
        def _impute_median_(c): xreg_df[c].fillna(xreg_df[c].median(),inplace=True)
        def _impute_mode_(c): xreg_df[c].fillna(stats.mode(xreg_df[c])[0][0],inplace=True)
        def _impute_min_(c): xreg_df[c].fillna(xreg_df[c].min(),inplace=True)
        def _impute_max_(c): xreg_df[c].fillna(xreg_df[c].max(),inplace=True)
        def _impute_0_(c): xreg_df[c].fillna(0,inplace=True)
        def _forward_fill_(c): xreg_df[c].fillna(method='ffill',inplace=True)
        def _backward_fill_(c): xreg_df[c].fillna(method='bfill',inplace=True)
        def _impute_random_(c): xreg_df.loc[xreg_df[c].isnull(),c] = xreg_df[c].dropna().sample(xreg_df.loc[xreg_df[c].isnull()].shape[0]).to_list()

        assert xreg_df.shape[0] == len(xreg_df[date_col].unique()), 'each date supplied must be unique'
        xreg_df[date_col] = pd.to_datetime(xreg_df[date_col])
        self.future_dates = xreg_df.loc[xreg_df[date_col] > list(self.current_dates)[-1],date_col].to_list()
        xreg_df = xreg_df.loc[xreg_df[date_col] >= self.current_dates[0]]
        xreg_df = pd.get_dummies(xreg_df,drop_first=True)

        if not not process_columns:
            if isinstance(process_columns,dict):
                for c, v in process_columns.items():
                    try: locals()['_'+v+'_'](c) if xreg_df[c].isnull().sum() > 0 else None
                    except KeyError: raise ValueError(f'argument {v} not supported for key {c} in process_columns')
            elif isinstance(process_columns,str):
                for c in xreg_df:
                    try: locals()['_'+process_columns+'_'](c) if xreg_df[c].isnull().sum() > 0 else None
                    except KeyError: raise ValueError(f'argument passed to process_columns not supported: {process_columns}')
            else:
                raise ValueError(f'argument passed to process_columns not supported: {process_columns}')

        current_xreg_df = xreg_df.loc[xreg_df[date_col].isin(self.current_dates)].drop(columns=date_col)
        future_xreg_df = xreg_df.loc[xreg_df[date_col] > list(self.current_dates)[-1]].drop(columns=date_col)        

        assert current_xreg_df.shape[0] == len(self.y), 'something went wrong--make sure the dataframe spans the entire date-range as y and is at least one observation to the future and specify a date column in date_col parameter'
        self.forecast_out_periods = future_xreg_df.shape[0]
        self.current_xreg = current_xreg_df.to_dict(orient='list')
        self.future_xreg = future_xreg_df.to_dict(orient='list')

    def generate_future_dates(self,n,freq):
        """ generates future dates and stores in the future_dates attribute
            changes forecast_out_periods attribute appropriately
            Parameters: n : int
                            the number of periods to forecast
                            the length of the resulting future_dates attribute
                        freq : str
                            a pandas datetime freq value
                            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        self.future_dates = pd.date_range(start=self.current_dates[-1],periods=n+1,freq=freq).to_list()[1:]
        self.set_forecast_out_periods(n)
        
    def set_forecast_out_periods(self,n):
        """ sets the self.forecast_out_periods attribute and truncates self.future_dates and self.future_xreg if needed
            Parameters: n : int
                            the number of periods you want to forecast out for
                            if this is a larger value than the size of the future_dates attribute, some models may fail
                            if this is a smaller value the the size of the future_dates attribute, future_xreg and future_dates will be truncated
        """
        if isinstance(n,int):
            if n >= 1:
                self.forecast_out_periods = n
                if (self.future_dates is None) | (n > len(self.future_dates)):
                    raise ForecastFormatError(f'cannot set forecast_out_periods to {n} without a fully populated future_dates attribute--try using the generate_future_dates() method')
                else:
                    self.future_dates = self.future_dates[:n]
                if isinstance(self.future_xreg,dict):
                    for k,v in self.future_xreg.items():
                        self.future_xreg[k] = v[:n]
            else:
                raise ValueError(f'n must be greater than 1, got {n}')  
        else:
            raise ValueError(f'n must be an int type, got {type(n)}')

    def set_ordered_xreg(self,chop_tail_periods=0,include_only='all',exclude=None,quiet=True):
        """ method for ordering stored externals from most to least correlated, according to absolute Pearson correlation coefficient value
            will not error out if a given external has no variation in it -- will simply skip
            when measuring correlation, will log/difference variables when possible to compare stationary results
            stores the results in self.ordered_xreg as a list
            if two vars are perfectly correlated, will skip the second one
            resuting self.ordered_xreg attribute may therefore not contain all xregs but will contain as many as could be set
            Parameters: chop_tail_periods : int, default 0
                            The number of periods to chop (to compare to a training dataset)
                            This is used to reduce the chance of overfitting the data by using mismatched test periods for forecasts
                        include_only : list or any other data type, default "all"
                            if this is a list, only the externals in the list will be considered when testing correlation
                            if this is not a list, then it will be ignored and all externals will be considered
                            if this is a list, exclude will be ignored
                        exclude : list or any other data type, default None
                            if this is a list, the externals in the list will be excluded when testing correlation
                            if this is not a list, then it will be ignored and no externals will be excluded
                            if include_only is a list, this is ignored
                            note: it is possible for include_only to be its default value, "all", and exclude to not be ignored if it is passed as a list type
                        quiet : bool, default True
                            if this is True, then if a given external is ignored (either because no correlation could be calculated or there are no observations after its tail has been chopped), you will not know
                            if this is False, then if a given external is ignored, it will print which external is being skipped
            >>> f = Forecaster()
            >>> f.get_data_fred('UTUR')
            >>> f.process_xreg_df(xreg_df,chop_tail_periods=12)
            >>> f.set_ordered_xreg()
            >>> print(f.ordered_xreg)
            ['x2','x1'] # in this case x2 correlates more strongly than x1 with y on a test set with 12 holdout periods
        """
        log_diff = lambda x: np.diff(np.log(x),n=1)

        if isinstance(include_only,list):
            use_these_externals = {}
            for e in include_only:
                use_these_externals[e] = self.current_xreg[e]
        else:
            use_these_externals = self.current_xreg.copy()
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
                y = np.array(self.y[:])
                
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
                r_coeff = stats.pearsonr(y,x)
            
            if np.abs(r_coeff[0]) not in ext_reg.values():
                ext_reg[k] = np.abs(r_coeff[0])
        
        k = Counter(ext_reg) 
        self.ordered_xreg = [h[0] for h in k.most_common()] # this should give us the ranked external regressors

    def forecast(self,which,**kwargs):
    	""" forecasts with a str method to allow for loops
    	"""
    	getattr(self,'forecast_'+which)(**kwargs)

    def forecast_auto_arima(self,test_length=1,Xvars=None,call_me='auto_arima'):
        """ Auto-Regressive Integrated Moving Average 
            forecasts using auto.arima from the forecast package in R
            uses an algorithm to find the best ARIMA model automatically by minimizing in-sample aic
            does not search seasonal models
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                            if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation
                            because the auto.arima function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option
                            if no arima model can be estimated, will raise an error
                        call_me : str, default "auto_arima"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            f = Forecaster()
            f.get_data_fred('UTUR')
            f.forecast_auto_arima(test_length=12,call_me='arima')
            print(f.info['arima'])
            >>> {'holdout_periods': 12, 
            >>> 'model_form': 'ARIMA(0,1,5)',
            >>> 'test_set_actuals': [2.4, 2.4, ..., 5.0, 4.1],
            >>> 'test_set_predictions': [2.36083282553252, 2.3119957980461803, ..., 2.09177057271149, 2.08127132827637], 
            >>> 'test_set_ape': [0.0163196560281154, 0.03666841748076, ..., 0.581645885457702, 0.49237284676186205]}
            print(f.forecasts['arima'])
            >>> [4.000616524942799, 4.01916650578768, ..., 3.7576542462753904, 3.7576542462753904]
            
            print(f.mape['arima'])
            >>> 0.4082393522799069
            print(f.feature_importance['arima']) # stored as a pandas dataframe
            >>>     coef        se    tvalue          pval
            >>> ma5  0.189706  0.045527  4.166858  3.598788e-05
            >>> ma4 -0.032062  0.043873 -0.730781  4.652316e-01
            >>> ma3 -0.060743  0.048104 -1.262753  2.072261e-01
            >>> ma2 -0.257684  0.044522 -5.787802  1.213441e-08
            >>> ma1  0.222933  0.042513  5.243861  2.265347e-07
        """
        self._ready_for_forecast()
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=Xvars)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            h <- {self.forecast_out_periods}
            data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
            data_train <- data[1:(nrow(data)-{test_length}),,drop=FALSE]
            data_test <- data[(nrow(data)-{test_length} + 1):nrow(data),,drop=FALSE]
            
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
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- auto.arima(y,max.order=10,stepwise=F,xreg=xreg_c)
            f <- forecast(ar,xreg=xreg_f,h=h)
            p <- f[[4]]
            arima_form <- f[[1]]
            
            write <- data.frame(forecast=p)
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
            write <- data.frame(fitted = fitted(ar))
            write.csv(write,'tmp/tmp_fitted.csv',row.names=F)

            summary_df = data.frame(coef=rev(coef(ar)),se=rev(sqrt(diag(vcov(ar)))))
            if (exists('externals')){row.names(summary_df)[1:length(externals)] <- externals}
            summary_df$tvalue = summary_df$coef/summary_df$se
            write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        tmp_fitted = pd.read_csv('tmp/tmp_fitted.csv')

        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = tmp_test_results['actual'].to_list()
        self.info[call_me]['test_set_predictions'] = tmp_test_results['forecast'].to_list()
        self.info[call_me]['test_set_ape'] = tmp_test_results['APE'].to_list()
        self._metrics(call_me)
        self.info[call_me]['fitted_values'] = tmp_fitted['fitted'].to_list()
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

        if self.feature_importance[call_me].shape[0] > 0: # for the (0,i,0) model case
            self.feature_importance[call_me]['pval'] = stats.t.sf(np.abs(self.feature_importance[call_me]['tvalue']), len(self.y)-1)*2 # https://stackoverflow.com/questions/17559897/python-p-value-from-t-statistic
        else:
            self.feature_importance.pop(call_me)

    def forecast_auto_arima_seas(self,start='auto',interval=12,test_length=1,Xvars=None,call_me='auto_arima_seas'):
        """ Auto-Regressive Integrated Moving Average 
            forecasts using auto.arima from the forecast package in R
            searches seasonal models, but the algorithm isn't as complex as forecast_auto_arima() and is harder to set up
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        start : tuple of length 2 or "auto", default "auto"
                            1st element is the start year
                            2nd element is the start period in the appropriate interval
                            for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)
                            if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list 
                        interval : float, default 12
                            the number of periods in one season (365.25 for annual, 12 for monthly, etc.)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                            if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation
                            because the auto.arima function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option
                            if no arima model can be estimated, will raise an error
                        call_me : str, default "auto_arima_seas"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self._ready_for_forecast()
        if start == 'auto':
            try: start = tuple(np.array(str(self.current_dates[0]).split('-')[:2]).astype(int))
            except: raise ValueError('could not set start automatically, try passing argument manually')

        try:
            float(interval)
        except ValueError:
            raise ValueError(f'interval must be numeric type, got {type(interval)}')

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=Xvars)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            start_p <- c{start}
            interval <- {interval}
            test_length <- {test_length}
            h <- {self.forecast_out_periods}
            
            data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
            
            y <- ts(data$y,start=start_p,deltat=1/interval)
            y_train <- subset(y,start=1,end=nrow(data)-test_length)
            y_test <- subset(y,start=nrow(data)-test_length+1,end=nrow(data))
            
            """)

        ro.r("""
            if (ncol(data) > 1){
              future_externals = data.frame(read.csv('tmp/tmp_r_future.csv'))
              externals <- names(data)[2:ncol(data)]
              data_c <- data[,externals, drop=FALSE]
              data_f <- future_externals[,externals, drop=FALSE]
              all_externals_ts <- ts(rbind(data_c,data_f),start=start_p,deltat=1/interval)
              xreg_c <- subset(all_externals_ts,start=1,end=nrow(data))
              xreg_tr <- subset(all_externals_ts,start=1,end=nrow(data)-test_length)
              xreg_te <- subset(all_externals_ts,start=nrow(data)-test_length+1,end=nrow(data))
              if (test_length == 1){
                xreg_te <- t(xreg_te)
              }
              xreg_f <- subset(all_externals_ts,start=nrow(data)+1)
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

            if (test_length==1){
              write <- data.frame(actual=y_test[1],
                                forecast=p[1])
            } else {
              write <- data.frame(actual=y_test,
                                forecast=p)            
            }            
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- auto.arima(y,xreg=xreg_c)
            f <- forecast(ar,xreg=xreg_f,h=h)
            p <- f[[4]]
            arima_form <- f[[1]]
            
            write <- data.frame(forecast=p)
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)

            write <- data.frame(fitted = fitted(ar))
            write.csv(write,'tmp/tmp_fitted.csv',row.names=F)

            summary_df = data.frame(coef=rev(coef(ar)),se=rev(sqrt(diag(vcov(ar)))))
            if (exists('externals')){row.names(summary_df)[1:length(externals)] <- externals}
            summary_df$tvalue = summary_df$coef/summary_df$se
            write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        tmp_fitted = pd.read_csv('tmp/tmp_fitted.csv')

        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = tmp_test_results['actual'].to_list()
        self.info[call_me]['test_set_predictions'] = tmp_test_results['forecast'].to_list()
        self.info[call_me]['test_set_ape'] = tmp_test_results['APE'].to_list()
        self._metrics(call_me)
        self.info[call_me]['fitted_values'] = tmp_fitted['fitted'].to_list()
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

        if self.feature_importance[call_me].shape[0] > 0: # for the (0,i,0) model case
            self.feature_importance[call_me]['pval'] = stats.t.sf(np.abs(self.feature_importance[call_me]['tvalue']), len(self.y)-1)*2 # https://stackoverflow.com/questions/17559897/python-p-value-from-t-statistic
        else:
            self.feature_importance.pop(call_me)

    def forecast_prophet(self,test_length=1,Xvars=None,call_me='prophet',**kwargs):
        """ Facebook Prophet
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                        call_me : str, default "prophet"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        key words are passed to Prophet() function
        """
        from fbprophet import Prophet

        self._ready_for_forecast()
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        if not Xvars is None:
            if Xvars == 'all':
                X = pd.DataFrame(self.current_xreg)
                X_f = pd.DataFrame(self.future_xreg)
            elif isinstance(Xvars,list):
                X = pd.DataFrame(self.current_xreg).loc[:,Xvars]
                X_f = pd.DataFrame(self.future_xreg).loc[:,Xvars]
            elif Xvars.startswith('top_'):
                nxreg = int(Xvars.split('_')[1])
                self.set_ordered_xreg(chop_tail_periods=test_length)
                X = pd.DataFrame(self.current_xreg).loc[:,self.ordered_xreg[:nxreg]]
                X_f = pd.DataFrame(self.future_xreg).loc[:,self.ordered_xreg[:nxreg]]
            else:
                raise ValueError(f'Xvars argument not recognized: {Xvars}')
        else:
            X = pd.DataFrame()
            X_f = pd.DataFrame()

        if 'cap' in kwargs:
            X['cap'] = [kwargs['cap']]*len(self.y)
            X_f['cap'] = [kwargs['cap']]*self.forecast_out_periods
            kwargs.pop('cap')

        if 'floor' in kwargs:
            X['floor'] = [kwargs['floor']]*len(self.y)
            X_f['floor'] = [kwargs['floor']]*self.forecast_out_periods
            kwargs.pop('floor')

        X['y'] = self.y
        X['ds'] = self.current_dates
        X_f['ds'] = self.future_dates

        # train/test
        model = Prophet(**kwargs)
        for x in X.iloc[:,:-2].columns:
            if x not in ('cap','floor'):
                model.add_regressor(x)
        model.fit(X.iloc[:-test_length])
        test = model.predict(X.iloc[-test_length:])
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = 'FB Prophet'
        self.info[call_me]['test_set_actuals'] = X.iloc[-test_length:,-2].to_list()
        self.info[call_me]['test_set_predictions'] = test['yhat'].to_list()
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(self.info[call_me]['test_set_predictions'],self.info[call_me]['test_set_actuals'])]
        self._metrics(call_me)

        # forecast
        model = Prophet(**kwargs)
        for x in X.iloc[:,:-2].columns:
            if x not in ('cap','floor'):
                model.add_regressor(x)
        model.fit(X)
        pred = model.predict(X_f)
        self.info[call_me]['fitted_values'] = model.predict(X)['yhat'].to_list()
        self.forecasts[call_me] = pred['yhat'].to_list()
        if X.shape[1] > 2:
            self.feature_importance[call_me] = pd.DataFrame(index=X.columns.to_list()[:-2])

    def forecast_sarimax13(self,test_length=1,start='auto',interval=12,Xvars=None,call_me='sarimax13',error='raise'):
        """ Seasonal Auto-Regressive Integrated Moving Average - ARIMA-X13 - https://www.census.gov/srd/www/x13as/
            Forecasts using the seas function from the seasonal package, also need the X13 software (x13as.exe) saved locally
            Automatically takes the best model ARIMA model form that fulfills a certain set of criteria (low forecast error rate, high statistical significance, etc)
            X13 is a sophisticated way to model seasonality with ARIMA maintained by the census bureau, and the seasonal package provides a simple wrapper around the software with R
            The function here is simplified, but the power in X13 is its database offers precise ways to model seasonality, also takes into account outliers
            Documentation: https://cran.r-project.org/web/packages/seasonal/seasonal.pdf, http://www.seasonal.website/examples.html
            This package only allows for monthly or less granular observations, and only three years or fewer of predictions
            the model will fail if there are 0 or negative values in the dependent variable attempted to be predicted
            the model can fail for several other reasons (including lack of seasonality in the dependent variable)
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        start : tuple of length 2 or "auto", default "auto"
                            1st element is the start year
                            2nd element is the start period in the appropriate interval
                            for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)
                            if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list 
                        interval : 1 of {1,2,4,12}, default 12
                            1 for annual, 2 for bi-annual, 4 for quarterly, 12 for monthly
                            unfortunately, x13 does not allow for more granularity than the monthly level
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                            if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation
                            because the seas function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option
                            x13 already has an extensive list of x regressors that it will pull automatically--read the documentation for more info
                        call_me : str, default "sarimax13"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        error: one of {"raise","pass","print"}, default "raise"
                            if unable to estimate the model, "raise" will raise an error
                            if unable to estimate the model, "pass" will silently skip the model and delete all associated attribute keys (self.info)
                            if unable to estimate the model, "print" will skip the model, delete all associated attribute keys (self.info), and print the error
                            errors are common even if you specify everything correctly -- it has to do with the X13 estimator itself
                            one common error is caused when negative or 0 values are present in the dependent variables
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self._ready_for_forecast()

        if min(self.y) <= 0:
            if error == 'raise':
                raise ValueError('cannot estimate nnetar model, negative or 0 values observed in the y attribute')
            elif error == 'pass':
                return None
            elif error == 'print':
                print('cannot estimate nnetar model, negative or 0 values observed in the y attribute')
                return None
            else:
                raise ValueError(f'argument in error not recognized: {error}')

        if start == 'auto':
            try: start = tuple(np.array(str(self.current_dates[0]).split('-')[:2]).astype(int))
            except: raise ValueError('could not set start automatically, try passing argument manually')

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        self._prepr('forecast','seasonal',test_length=test_length,call_me=call_me,Xvars=Xvars)

        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            start_p <- c{start}
            interval <- {interval}
            test_length <- {test_length}
            
            data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
            
            y <- ts(data$y,start=start_p,deltat=1/interval)
            y_train <- subset(y,start=1,end=nrow(data)-test_length)
            y_test <- subset(y,start=nrow(data)-test_length+1,end=nrow(data))
            
        """)

        ro.r("""
            if (ncol(data) > 1){
              future_externals = data.frame(read.csv('tmp/tmp_r_future.csv'))
              r <- max(0,36-nrow(future_externals))
              filler <-as.data.frame(replicate(ncol(future_externals),rep(0,r))) # we always need at least three years of data for this package
              # if we have less than three years, fill in the rest with 0s
              # we still only use predictions matching whatever is stored in self.forecast_out_periods
              # https://github.com/christophsax/seasonal/issues/200
              names(filler) <- names(future_externals)
              future_externals <- rbind(future_externals,filler)
              externals <- names(data)[2:ncol(data)]
              data_c <- data[,externals, drop=FALSE]
              data_f <- future_externals[,externals, drop=FALSE]
              all_externals_ts <- ts(rbind(data_c,data_f),start=start_p,deltat=1/interval)
            } else {
              all_externals_ts <- NULL
            }
        """)

        try:
            ro.r(f"""
                    m_test <- seas(x=y_train,xreg=all_externals_ts,forecast.save="forecasts",pickmdl.method="best")
                    p <- series(m_test, "forecast.forecasts")[1:test_length,]
                    m <- seas(x=y,xreg=all_externals_ts,forecast.save="forecasts",pickmdl.method="best")
                    f <- series(m, "forecast.forecasts")[1:{self.forecast_out_periods},]
                    arima_form <- paste('ARIMA-X13',m_test$model$arima$model)
                    write <- data.frame(actual=y_test,forecast=p[,1])
                    write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
                    write$model_form <- arima_form
                    write.csv(write,'tmp/tmp_test_results.csv',row.names=F)
                    arima_form <- paste('ARIMA-X13',m$model$arima$model)
                    write <- data.frame(forecast=f[,1])
                    write$model_form <- arima_form
                    write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
            """)
        except Exception as e:
            self.info.pop(call_me)
            if error == 'raise':
                raise e
            elif error == 'print':
                print(f"skipping model, here's the error:\n{e}")
            elif error != 'pass':
                print(f'error argument not recognized: {error}')
                raise e
            return None

        ro.r("""
            # feature_importance -- cool output
            summary_df <- data.frame(summary(m))
            if (exists("externals")) {summary_df$term[1:length(externals)] <- externals}
            write.csv(summary_df,'tmp/tmp_summary_output.csv',row.names=F)
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.forecasts[call_me] = tmp_forecast['forecast'].to_list()
    
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = tmp_test_results['actual'].to_list()
        self.info[call_me]['test_set_predictions'] = tmp_test_results['forecast'].to_list()
        self.info[call_me]['test_set_ape'] = tmp_test_results['APE'].to_list()
        self._metrics(call_me)
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

    def forecast_arima(self,test_length=1,Xvars=None,order=(0,0,0),seasonal_order=(0,0,0,0),trend=None,call_me='arima',**kwargs):
        """ ARIMA model from statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
            the args endog, exog, and dates passed automatically
            the args order, seasonal_order, and trend should be specified in the method
            all other arguments in the ARIMA() function can be passed to kwargs
            using this framework, the following model types can be specified:
                AR, MA, ARMA, ARIMA, SARIMA, regression with ARIMA errors
            this is meant for manual arima modeling; for a more automated implementation, see the forecast_auto_arima() and forecast_sarimax13() methods
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", or None default None
                            the independent variables to use in the resulting X dataframes
                            "top_" not supported
                        call_me : str, default "arima"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        Info about all other arguments (order, seasonal_order, trend) can be found in the sm.tsa.arima.model.ARIMA documentation (linked above)
                        other arguments from ARIMA() function can be passed as keywords
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from statsmodels.tsa.arima.model import ARIMA

        self._ready_for_forecast()
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        if not Xvars is None:
            if Xvars == 'all':
                X = pd.DataFrame(self.current_xreg)
                X_f = pd.DataFrame(self.future_xreg)
            elif isinstance(Xvars,list):
                X = pd.DataFrame(self.current_xreg).loc[:,Xvars]
                X_f = pd.DataFrame(self.future_xreg).loc[:,Xvars]
            else:
                raise ValueError(f'Xvars argument not recognized: {Xvars}')
        else:
            X = None
            X_f = None

        y = pd.Series(self.y)

        X_train = None if Xvars is None else X.iloc[:-test_length]
        X_test = None if Xvars is None else X.iloc[-test_length:]
        y_train = y.values[:-test_length]
        y_test = y.values[-test_length:]
        dates = pd.to_datetime(self.current_dates) if not self.current_dates is None else None

        arima_train = ARIMA(y_train,exog=X_train,order=order,seasonal_order=seasonal_order,trend=trend,dates=dates,**kwargs).fit()
        pred = list(arima_train.predict(exog=X_test,start=len(y_train),end=len(y)-1,typ='levels'))
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = 'ARIMA {}x{} include {}'.format(order,seasonal_order,trend)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = pred
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(pred,y_test)]

        self._metric()

        arima = ARIMA(y,exog=X,order=order,seasonal_order=seasonal_order,trend=trend,dates=dates,**kwargs).fit()
        self.info[call_me]['fitted_values'] = list(arima.fittedvalues)
        self.forecasts[call_me] = list(arima.predict(exog=X_f,start=len(y),end=len(y) + self.forecast_out_periods-1,typ='levels'))
        self._sm_summary_to_fi(arima,call_me)

    def forecast_hwes(self,test_length=1,call_me='hwes',**kwargs):
        """ Holt-Winters Exponential Smoothing
            https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
            https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572
            The Holt-Winters ES modifies the Holt ES technique so that it can be used in the presence of both trend and seasonality.
            for a more automated holt-winters application, see forecast_auto_hwes()
            if no keywords are added, this is almost always the same as a naive forecast that propogates the final value forward
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        call_me : str, default "hwes"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        keywords are passed to the ExponentialSmoothing function from statsmodels -- `dates` is specified automatically
                        some important parameters to specify as key words: trend, damped_trend, seasonal, seasonal_periods, use_boxcox
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

        self._ready_for_forecast()
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()

        y = pd.Series(self.y)
        y_train = y.values[:-test_length]
        y_test = y.values[-test_length:]
        dates = pd.to_datetime(self.current_dates) if not self.current_dates is None else None

        hwes_train = HWES(y_train,dates=dates,**kwargs).fit(optimized=True,use_brute=True)
        pred = list(hwes_train.predict(start=len(y_train),end=len(y)-1))
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = 'Holt-Winters Exponential Smoothing {}'.format(kwargs)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = pred
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(pred,y_test)]
        self._metrics(call_me)

        hwes = HWES(y,dates=dates,**kwargs).fit(optimized=True,use_brute=True)
        self.info[call_me]['fitted_values'] = list(hwes.fittedvalues)
        self.forecasts[call_me] = list(hwes.predict(start=len(y),end=len(y) + self.forecast_out_periods-1))
        self._sm_summary_to_fi(hwes,call_me)

    def forecast_auto_hwes(self,test_length=1,seasonal=False,seasonal_periods=None,call_me='auto_hwes'):
        """ Holt-Winters Exponential Smoothing
            https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
            https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572
            The Holt-Winters ES modifies the Holt ES technique so that it can be used in the presence of both trend and seasonality.
            Will add different trend and seasonal components automatically and test which minimizes AIC of in-sample predictions
            uses optimized model parameters to fit final model
            for a more manual holt-winters application, see forecast_hwes()
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        seasonal : bool, default False
                            whether there is seasonality in the series
                        seasonal_periods : int, default None
                            the number of periods to complete one seasonal period (for monthly, this is 12)
                            ignored if seasonal is False
                        call_me : str, default "hwes"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
        from itertools import product

        self._ready_for_forecast()
        expand_grid = lambda d: pd.DataFrame([row for row in product(*d.values())],columns=d.keys())

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        if seasonal:
            assert isinstance(seasonal_periods,int),'seasonal_periods must be int type when seasonal is True'
            assert seasonal_periods > 0,'seasonal_periods must be greater than 1 when seasonal is True'
        elif not seasonal:
            seasonal_periods = None
        else:
            raise ValueError(f'argument passed to seasonal not recognized: {seasonal}')

        self.info[call_me] = self._get_info_dict()

        y = pd.Series(self.y)
        y_train = y.values[:-test_length]
        y_test = y.values[-test_length:]
        dates = pd.to_datetime(self.current_dates)

        scores = [] # lower is better
        if y.min() > 0:
            grid = expand_grid({
                'trend':[None,'add','mul'],
                'seasonal':[None] if not seasonal else ['add','mul'],
                'damped_trend':[True,False],
                'use_boxcox':[True,False,0], # the last one is a log transformation
                'seasonal_periods':[None] if not seasonal else [seasonal_periods]
            })
        else: # multiplicative trends only work for series of all positive figures
            grid = expand_grid({
                'trend':[None,'add'],
                'seasonal':[None] if not seasonal else ['add'],
                'damped_trend':[True,False],
                'use_boxcox':[None], # the last one is a log transformation
                'seasonal_periods':[None] if not seasonal else [seasonal_periods]
            })

        grid = grid.loc[((grid['trend'].isnull()) & (~grid['damped_trend'])) | (~grid['trend'].isnull())].reset_index(drop=True) # it does not know how to damp when there is no trend

        for i, params in grid.iterrows():
            hwes_scored = HWES(y_train,dates=dates,initialization_method='estimated',**params).fit(optimized=True,use_brute=True)
            scores.append(hwes_scored.aic)

        grid['aic'] = scores
        best_params = grid.dropna(subset=['aic']).loc[grid['aic'] == grid['aic'].min()].drop(columns='aic').iloc[0]

        hwes_train = HWES(y_train,dates=dates,initialization_method='estimated',**best_params).fit(optimized=True,use_brute=True)
        pred = list(hwes_train.predict(start=len(y_train),end=len(y)-1))
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = 'Holt-Winters Exponential Smoothing {}'.format(dict(best_params))
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = pred
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(pred,y_test)]
        self._metrics(call_me)

        hwes = HWES(y,dates=dates,initialization_method='estimated',**best_params).fit(optimized=True,use_brute=True)
        self.info[call_me]['fitted_values'] = list(hwes.fittedvalues)
        self.forecasts[call_me] = list(hwes.predict(start=len(y),end=len(y) + self.forecast_out_periods-1))
        self._sm_summary_to_fi(hwes,call_me)

    def forecast_nnetar(self,test_length=1,start='auto',interval=12,Xvars=None,P=1,boxcox=False,scale_inputs=True,repeats=20,negative_y='raise',call_me='nnetar'):
        """ Neural Network Time Series Forecast
            uses nnetar function from the forecast package in R
            this forecast does not work when there are negative or 0 values in the dependent variable
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        start : tuple of length 2 or "auto", default "auto"
                            1st element is the start year
                            2nd element is the start period in the appropriate interval
                            for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)
                            if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list 
                        interval : float, default 12
                            the number of periods in one season (365.25 for annual, 12 for monthly, etc.)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                            if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation
                            because the function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option
                            if no model can be estimated, will raise an error
                        P : int, default 1
                            the number of seasonal lags to add to the model
                        boxcox : bool, default False
                            whether to use a boxcox transformation on y
                        scale_inputs : bool, default True
                            whether to scale the inputs, performed after the boxcox transformation if that is set to True
                        repeats : int, default 20
                            the number of models to average with different starting points
                        negative_y : one of {'raise','pass','print'}, default 'raise'
                            what to do if negative or 0 values are observed in the y attribute
                            'raise' will raise a ValueError
                            'pass' will not attempt to evaluate a model without raising an error
                            'print' will not evaluate the model but print the error
                        call_me : str, default "nnetar"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
        """
        self._ready_for_forecast()

        if min(self.y) <= 0:
            if negative_y == 'raise':
                raise ValueError('cannot estimate nnetar model, negative or 0 values observed in the y attribute')
            elif negative_y == 'pass':
                return None
            elif negative_y == 'skip':
                print('cannot estimate nnetar model, negative or 0 values observed in the y attribute')
                return None
            else:
                raise ValueError(f'argument in negative_y not recognized: {negative_y}')

        if start == 'auto':
            try: start = tuple(np.array(str(self.current_dates[0]).split('-')[:2]).astype(int))
            except: raise ValueError('could not set start automatically, try passing argument manually')

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        assert isinstance(repeats,int), f'repeats must be an int, not {type(repeats)}'

        if isinstance(boxcox,bool):
            bc = str(boxcox)[0]
        else:
            raise ValueError(f'argument passed to boxcox not recognized: {boxcox}')

        if isinstance(scale_inputs,bool):
            si = str(scale_inputs)[0]
        else:
            raise ValueError(f'argument passed to scale_inputs not recognized: {scale_inputs}')

        try:
            float(interval)
        except ValueError:
            raise ValueError(f'interval must be numeric type, got {type(interval)}')

        self.info[call_me] = self._get_info_dict()
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=Xvars)

        ro.r(f"""
            rm(list=ls())
            set.seed(20)
            setwd('{rwd}')
            start_p <- c{start}
            interval <- {interval}
            test_length <- {test_length}
            P <- {P}
            boxcox <- {bc}
            scale_inputs <- {si}
            repeats <- {repeats}
            h <- {self.forecast_out_periods}
            
            data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
            
            y <- ts(data$y,start=start_p,deltat=1/interval)
            y_train <- subset(y,start=1,end=nrow(data)-test_length)
            y_test <- subset(y,start=nrow(data)-test_length+1,end=nrow(data))
            
            """)

        ro.r("""
            if (ncol(data) > 1){
              future_externals = data.frame(read.csv('tmp/tmp_r_future.csv'))
              externals <- names(data)[2:ncol(data)]
              data_c <- data[,externals, drop=FALSE]
              data_f <- future_externals[,externals, drop=FALSE]
              all_externals_ts <- ts(rbind(data_c,data_f),start=start_p,deltat=1/interval)
              xreg_c <- subset(all_externals_ts,start=1,end=nrow(data))
              xreg_tr <- subset(all_externals_ts,start=1,end=nrow(data)-test_length)
              xreg_te <- subset(all_externals_ts,start=nrow(data)-test_length+1,end=nrow(data))
              if (test_length == 1){
                xreg_te <- t(xreg_te)
              }
              xreg_f <- subset(all_externals_ts,start=nrow(data)+1)
            } else {
              xreg_c <- NULL
              xreg_tr <- NULL
              xreg_te <- NULL
              xreg_f <- NULL
            }
            ar <- nnetar(y_train,xreg=xreg_tr,lambda=boxcox,scale.inputs=scale_inputs,repeats=repeats,P=P)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            p <- f$mean
            nn_form <- f$method
            if (test_length==1){
              write <- data.frame(actual=y_test[1],
                                forecast=p[1])
            } else {
              write <- data.frame(actual=y_test,
                                forecast=p)            
            }

            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- nn_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- nnetar(y,xreg=xreg_c,lambda=boxcox,scale.inputs=scale_inputs,repeats=repeats,P=P)
            f <- forecast(ar,xreg=xreg_f,h=h)
            p <- f$mean
            nn_form <- f$method
            
            write <- data.frame(forecast=p)
            write$model_form <- nn_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)

            write <- data.frame(fitted = fitted(ar))
            write.csv(write,'tmp/tmp_fitted.csv',row.names=F)
        """)
        

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        tmp_fitted = pd.read_csv('tmp/tmp_fitted.csv')
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = tmp_test_results['actual'].to_list()
        self.info[call_me]['test_set_predictions'] = tmp_test_results['forecast'].to_list()
        self.info[call_me]['test_set_ape'] = tmp_test_results['APE'].to_list()
        self._metrics(call_me)
        self.info[call_me]['fitted_values'] = tmp_fitted['fitted'].to_list()
        self.feature_importance[call_me] = pd.DataFrame(index=pd.read_csv('tmp/tmp_r_current.csv').iloc[:,1:].columns.to_list())
        if self.feature_importance[call_me].shape[0] == 0:
            self.feature_importance.pop(call_me)

    def forecast_tbats(self,test_length=1,season='NULL',call_me='tbats'):
        """ Exponential Smoothing State Space Model With Box-Cox Transformation, ARMA Errors, Trend And Seasonal Component
            forecasts using tbats from the forecast package in R
            auto-regressive only (no external regressors)
            this is an automated model selection
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        season : int or "NULL"
                            the number of seasonal periods to consider (12 for monthly, etc.)
                            if no seasonality desired, leave "NULL" as this will be passed directly to the tbats function in r
                        call_me : str, default "tbats"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self._ready_for_forecast()
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'

        self.info[call_me] = self._get_info_dict()
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=None)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')
            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
        
            ar <- tbats(y_train,seasonal.periods={season})
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[2]] are point estimates, f[[9]] is the TBATS form
            p <- f[[2]]
            tbats_form <- f[[9]]
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- tbats_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)
            ar <- tbats(y)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[2]]
            tbats_form <- f[[9]]
            
            write <- data.frame(forecast=p)
            write$model_form <- tbats_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
            write <- data.frame(fitted = fitted(ar))
            write.csv(write,'tmp/tmp_fitted.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        tmp_fitted = pd.read_csv('tmp/tmp_fitted.csv')

        self.forecasts[call_me] = tmp_forecast['forecast'].to_list()
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = tmp_test_results['actual'].to_list()
        self.info[call_me]['test_set_predictions'] = tmp_test_results['forecast'].to_list()
        self.info[call_me]['test_set_ape'] = tmp_test_results['APE'].to_list()
        self._metrics(call_me)
        self.info[call_me]['fitted_values'] = tmp_fitted['fitted'].to_list()

    def forecast_ets(self,test_length=1,call_me='ets'):
        """ Exponential Smoothing State Space Model
            forecasts using ets from the forecast package in R
            auto-regressive only (no external regressors)
            this is an automated model selection
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        call_me : str, default "ets"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self._ready_for_forecast()
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'

        self.info[call_me] = self._get_info_dict()
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=None)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')
            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
        
            ar <- ets(y_train)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[2]] are point estimates, f[[8]] is the ETS form
            p <- f[[2]]
            ets_form <- f[[8]]
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- ets_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)
            ar <- ets(y)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[2]]
            ets_form <- f[[8]]
            
            write <- data.frame(forecast=p)
            write$model_form <- ets_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
            write <- data.frame(fitted = fitted(ar))
            write.csv(write,'tmp/tmp_fitted.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        tmp_fitted = pd.read_csv('tmp/tmp_fitted.csv')

        self.forecasts[call_me] = tmp_forecast['forecast'].to_list()
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = tmp_test_results['actual'].to_list()
        self.info[call_me]['test_set_predictions'] = tmp_test_results['forecast'].to_list()
        self.info[call_me]['test_set_ape'] = tmp_test_results['APE'].to_list()
        self._metrics(call_me)
        self.info[call_me]['fitted_values'] = tmp_fitted['fitted'].to_list()

    def forecast_var(self,*series,auto_resize=False,test_length=1,Xvars=None,lag_ic='AIC',optimizer='AIC',season='NULL',max_externals=None,call_me='var'):
        """ Vector Auto Regression
            forecasts using VAR from the vars package in R
            Optimizes the final model with different time trends, constants, and x variables by minimizing the AIC or BIC in the training set
            Unfortunately, only supports a level forecast, so to avoid stationarity issues, perform your own transformations before loading the data
            Parameters: series : required
                            lists of other series to run the VAR with
                            each list must be the same size as self.y if auto_resize is False
                            be sure to exclude NAs
                        auto_resize : bool, default False
                            if True, if series in series are different size than self.y, all series will be truncated to match the shortest series
                            if True, note that the forecast will not necessarily make predictions based on the entire history available in y
                            using this assumes that the shortest series ends at the same time the others do and there are no periods missing
                        test_length : int, default 1
                            the number of periods to hold out in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors
                            if it is "all", will attempt to estimate a model with all available x regressors
                            because the VAR function will fail if there is perfect collinearity in any of the xregs or if there is no variation in any of the xregs, using "top_" is safest option
                        lag_ic : str, one of {"AIC", "HQ", "SC", "FPE"}; default "AIC"
                            the information criteria used to determine the optimal number of lags in the VAR function
                        optimizer : str, one of {"AIC","BIC"}; default "AIC"
                            the information criteria used to select the best model in the optimization grid
                            a good, short resource to understand the difference: https://www.methodology.psu.edu/resources/AIC-vs-BIC/
                        season : int, default "NULL"
                            the number of periods to add a seasonal component to the model
                            if "NULL", no seasonal component will be added
                            don't use None ("NULL" is passed directly to the R CRAN mirror)
                            example: if your data is monthly and you suspect seasonality, you would make this 12
                        max_externals: int or None type, default None
                            the maximum number of externals to try in each model iteration
                            0 to this value of externals will be attempted and every combination of externals will be tried
                            None signifies that all combinations will be tried
                            reducing this from None can speed up processing and reduce overfitting
                        call_me : str, default "var"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self._ready_for_forecast()
        if len(series) == 0:
            raise ValueError('cannot run var -- need at least 1 series passed to *series')
        series_df = pd.DataFrame()
        min_size = min([len(s) for s in series] + [len(self.y)])
        for i, s in enumerate(series):
            if not isinstance(s,list):
                raise TypeError(f'cannot run var -- not a list type ({type(s)}) passed to *series')
            elif (len(s) != len(self.y)) & (not auto_resize):
                raise ValueError('cannot run var -- at least 1 list passed to *series is different length than y--try changing auto_resize to True')
            elif auto_resize:
                s = s[(len(s) - min_size):]
            elif not not auto_resize:
                raise ValueError(f'argument in auto_resize not recognized: {auto_resize}')
            series_df[f'cid{i+1}'] = s

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        assert optimizer in ('AIC','BIC'), f'cannot estimate model - optimizer value of {optimizer} not recognized'
        assert lag_ic in ("AIC", "HQ", "SC", "FPE"), f'cannot estimate model - lag_ic value of {lag_ic} not recognized'

        if (max_externals is None) & (not Xvars is None):
            if isinstance(Xvars,list):
                max_externals = len(Xvars)
            elif Xvars == 'all':
                max_externals = len(self.current_xreg.keys())
            elif Xvars.startswith('top_'):
                max_externals = int(Xvars.split('_')[1])
            else:
                raise ValueError(f'Xvars argument {Xvars} not recognized')

        self._prepr('vars',test_length=test_length,call_me=call_me,Xvars=Xvars)
        series_df.to_csv('tmp/tmp_r_cid.csv',index=False)
        self.info[call_me] = self._get_info_dict()

        ro.r(f"""
                rm(list=ls())
                setwd('{rwd}')
                data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
                cid <- na.omit(data.frame(read.csv('tmp/tmp_r_cid.csv')))
                test_periods <- {test_length}
                max_ext <- {max_externals if not max_externals is None else 0}
                lag_ic <- "{lag_ic}"
                IC <- {optimizer}
                season <- {season}
                n_ahead <- {self.forecast_out_periods}
            """)

        ro.r("""
                total_length <- min(nrow(data),nrow(cid))
                data <- data[(nrow(data) - total_length + 1) : nrow(data),,drop=FALSE]
                cid <- cid[(nrow(cid) - total_length + 1) : nrow(cid),,drop=FALSE]

                if (ncol(data) > 1){
                    exogen_names <- names(data)[2:ncol(data)]
                    exogen_future <- read.csv('tmp/tmp_r_future.csv')
                    exogen_train <- data[1:(nrow(data)-test_periods),names(data)[2:ncol(data)], drop=FALSE]
                    exogen_test <- data[(nrow(data)-(test_periods-1)):nrow(data),names(data)[2:ncol(data)], drop=FALSE]

                    # every combination of the external regressors, including no external regressors
                    exogenstg1 <- list()
                    for (i in 1:length(exogen_names)){
                      exogenstg1[[i]] <- combn(exogen_names,i)
                    }

                    h <- 2
                    exogen <- list(NULL)
                    for (i in 1:min(max_ext,length(exogenstg1))) {
                      for (j in 1:ncol(exogenstg1[[i]])) {
                        exogen[[h]] <- exogenstg1[[i]][,j]
                        h <- h+1
                      }
                    }
                } else {
                    exogen <- list(NULL)
                    exogen_future <- NULL
                }
                
                data.ts <- cbind(data[[1]],cid)
                data.ts_train <- data.ts[1:(nrow(data)-test_periods),,drop=FALSE]
                data.ts_test <- data.ts[(nrow(data)-(test_periods-1)):nrow(data),,drop=FALSE]

                # create a grid of parameters for the best estimator for each series pair
                include = c('none','const','trend','both')

                grid <- expand.grid(include = include, exogen=exogen)
                grid$ic <- 999999

                for (i in 1:nrow(grid)){
                  if (is.null(grid[i,'exogen'][[1]])){
                    ex_train = NULL
                  } else {
                    ex_train = exogen_train[,grid[i,'exogen'][[1]]]
                  }

                  vc_train <- VAR(data.ts_train,
                                      season=season,
                                      ic='AIC',
                                      type=as.character(grid[i,'include']),
                                      exogen=ex_train)
                  grid[i,'ic'] <-  IC(vc_train)
                }

                # choose parameters with best IC
                best_params <- grid[grid$ic == min(grid$ic),]

                # set externals
                if (is.null(best_params[1,'exogen'][[1]])){
                  ex_current = NULL
                  ex_future = NULL
                  ex_train = NULL
                  ex_test = NULL

                } else {
                  ex_current = as.matrix(data[,best_params[1,'exogen'][[1]]])
                  ex_future = as.matrix(exogen_future[,best_params[1,'exogen'][[1]]])
                  ex_train = as.matrix(exogen_train[,best_params[1,'exogen'][[1]]])
                  ex_test = as.matrix(exogen_test[,best_params[1,'exogen'][[1]]])
                }
                
                # predict on test set one more time with best parameters for model accuracy info
                vc_train <- VAR(data.ts_train,
                                 season=season,
                                 ic='AIC',
                                 type = as.character(best_params[1,'include']),
                                 exogen=ex_train)
                pred <- predict(vc_train,n.ahead=test_periods,dumvar=ex_test)
                p <- data.frame(row.names=1:nrow(data.ts_test))
                for (i in 1:length(pred$fcst)) {
                  p$col <- pred$fcst[[i]][,1]
                  names(p)[i] <- paste0('series',i)
                }
                p$xreg <- as.character(best_params[1,'exogen'])[[1]]
                p$model_form <- paste0('VAR',' include: ',best_params[1,'include'],'|selected regressors: ',as.character(best_params[1,'exogen'])[[1]])

                write.csv(p,'tmp/tmp_test_results.csv',row.names=F)

                # train the final model on full dataset with best parameter values
                vc.out = VAR(data.ts,
                              season=season,
                              ic=lag_ic,
                              type = as.character(best_params[1,'include']),
                              exogen=ex_current
                )
                # make the forecast
                fcst <- predict(vc.out,n.ahead=n_ahead,dumvar=ex_future)
                f <- data.frame(row.names=1:n_ahead)
                for (i in 1:length(fcst$fcst)) {
                  f$col <- fcst$fcst[[i]][,1]
                  names(f)[i] <- paste0('series',i)
                }
                # write final forecast values
                write.csv(f,'tmp/tmp_forecast.csv',row.names=F)

                summary_df <- coef(vc_train)[[1]]
                write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')

        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['test_set_predictions'] = tmp_test_results.iloc[:,0].to_list()
        self.info[call_me]['test_set_actuals'] = self.y[(-test_length):]
        self.info[call_me]['test_set_ape'] = [np.abs(y - yhat) / np.abs(y) for y, yhat in zip(self.y[(-test_length):],tmp_test_results.iloc[:,0])]
        self.info[call_me]['model_form'] = tmp_test_results['model_form'][0]
        self._metrics(call_me)
        self.forecasts[call_me] = tmp_forecast.iloc[:,0].to_list()
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

    def forecast_vecm(self,*cids,auto_resize=False,test_length=1,Xvars=None,r=1,max_lags=6,optimizer='AIC',max_externals=None,call_me='vecm'):
        """ Vector Error Correction Model
            forecasts using VECM from the tsDyn package in R
            Optimizes the final model with different lags, time trends, constants, and x variables by minimizing the AIC or BIC in the training set
            Parameters: cids : required
                            lists of cointegrated data
                            each list must be the same size as self.y
                            if this is only 1 list, it must be cointegrated with self.y
                            if more than 1 list, there must be at least 1 cointegrated pair between cids* and self.y (to fulfill the requirements of VECM)
                            be sure to exclude NAs
                        auto_resize : bool, default False
                            if True, if series in cids are different size than self.y, all series will be truncated to match the shortest series
                            if True, note that the forecast will not necessarily make predictions based on the entire history available in y
                            using this assumes that the shortest series ends at the same time the others do and there are no periods missing
                        test_length : int, default 1
                            the number of periods to hold out in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors
                            if it is "all", will attempt to estimate a model with all available x regressors
                            because the VECM function will fail if there is perfect collinearity in any of the xregs or if there is no variation in any of the xregs, using "top_" is safest option
                        r : int, default 1
                            the number of total cointegrated relationships between self.y and cids
                            if not an int or less than 1, an AssertionError is raised
                        max_lags : int, default 6
                            the total number of lags that will be used in the optimization process
                            1 to this number will be attempted
                            if not an int or less than 0, an AssertionError is raised
                        optimizer : str, one of {"AIC","BIC"}; default "AIC"
                            the information criteria used to select the best model in the optimization grid
                            a good, short resource to understand the difference: https://www.methodology.psu.edu/resources/AIC-vs-BIC/
                        max_externals: int or None type, default None
                            the maximum number of externals to try in each model iteration
                            0 to this value of externals will be attempted and every combination of externals will be tried
                            None signifies that all combinations will be tried
                            reducing this from None can speed up processing and reduce overfitting
                        call_me : str, default "vecm"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self._ready_for_forecast()
        if len(cids) == 0:
            raise ValueError('cannot run vecm -- need at least 1 cointegrated series in a list that is same length as y passed to *cids--no list found')
        cid_df = pd.DataFrame()
        min_size = min([len(cid) for cid in cids] + [len(self.y)])
        for i, cid in enumerate(cids):
            if not isinstance(cid,list):
                raise TypeError('cannot run var -- need at least 1 series in a list passed to *cids--not a list type detected')
            elif (len(cid) != len(self.y)) & (not auto_resize):
                raise ValueError('cannot run var -- need at least 1 series in a list that is same length as y passed to *cids--at least 1 list is different length than y--try changing auto_resize to True')
            elif auto_resize:
                cid = cid[(len(cid) - min_size):]
            elif not not auto_resize:
                raise ValueError(f'argument in auto_resize not recognized: {auto_resize}')
            cid_df[f'cid{i+1}'] = cid

        assert isinstance(r,int), f'r must be an int, not {type(r)}'
        assert r >= 1, 'r must be at least 1'
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        assert isinstance(max_lags,int), f'max_lags must be an int, not {type(max_lags)}'
        assert max_lags >= 0, 'max_lags must be positive'
        assert optimizer in ('AIC','BIC'), f'cannot estimate model - optimizer value of {optimizer} not recognized'

        if (max_externals is None) & (not Xvars is None):
            if isinstance(Xvars,list):
                max_externals = len(Xvars)
            elif Xvars == 'all':
                max_externals = len(self.current_xreg.keys())
            elif Xvars.startswith('top_'):
                max_externals = int(Xvars.split('_')[1])
            else:
                raise ValueError(f'Xvars argument {Xvars} not recognized')

        self._prepr('tsDyn',test_length=test_length,call_me=call_me,Xvars=Xvars)
        cid_df.to_csv('tmp/tmp_r_cid.csv',index=False)
        self.info[call_me] = self._get_info_dict()

        ro.r(f"""
                rm(list=ls())
                setwd('{rwd}')
                data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
                cid <- data.frame(read.csv('tmp/tmp_r_cid.csv'))
                test_periods <- {test_length}
                r <- {r}
                IC <- {optimizer}
                max_ext <- {max_externals if not max_externals is None else 0}
                max_lags <- {max_lags}
                n_ahead <- {self.forecast_out_periods}
            """)

        ro.r("""
                total_length <- min(nrow(data),nrow(cid))
                data <- data[(nrow(data) - total_length + 1) : nrow(data),,drop=FALSE]
                cid <- cid[(nrow(cid) - total_length + 1) : nrow(cid),,drop=FALSE]

                if (ncol(data) > 1){
                    exogen_names <- names(data)[2:ncol(data)]
                    exogen_future <- read.csv('tmp/tmp_r_future.csv')
                    exogen_train <- data[1:(nrow(data)-test_periods),names(data)[2:ncol(data)], drop=FALSE]
                    exogen_test <- data[(nrow(data)-(test_periods-1)):nrow(data),names(data)[2:ncol(data)], drop=FALSE]

                    # every combination of the external regressors, including no external regressors
                    exogenstg1 <- list()
                    for (i in 1:length(exogen_names)){
                      exogenstg1[[i]] <- combn(exogen_names,i)
                    }

                    h <- 2
                    exogen <- list(NULL)
                    for (i in 1:min(max_ext,length(exogenstg1))) {
                      for (j in 1:ncol(exogenstg1[[i]])) {
                        exogen[[h]] <- exogenstg1[[i]][,j]
                        h <- h+1
                      }
                    }
                } else {
                    exogen <- list(NULL)
                    exogen_future <- NULL
                }
                                
                data.ts <- cbind(data[[1]],cid)
                data.ts_train <- data.ts[1:(nrow(data)-test_periods),]
                data.ts_test <- data.ts[(nrow(data)-(test_periods-1)):nrow(data),]

                # create a grid of parameters for the best estimator for each series pair
                lags = seq(1,max_lags)
                include = c('none','const','trend','both')
                estim = c('2OLS','ML')

                grid <- expand.grid(lags = lags, include = include, estim = estim, exogen=exogen)
                grid$ic <- 999999

                for (i in 1:nrow(grid)){
                  if (is.null(grid[i,'exogen'][[1]])){
                    ex_train = NULL
                  } else {
                    ex_train = exogen_train[,grid[i,'exogen'][[1]]]
                  }

                  vc_train <- VECM(data.ts_train,
                                  r=r,
                                  lag=grid[i,'lags'],
                                  include = as.character(grid[i,'include']),
                                  estim = as.character(grid[i,'estim']),
                                  exogen=ex_train)
                  grid[i,'ic'] <-  IC(vc_train)
                }

                # choose parameters with best IC
                best_params <- grid[grid$ic == min(grid$ic),]

                # set externals
                if (is.null(best_params[1,'exogen'][[1]])){
                  ex_current = NULL
                  ex_future = NULL
                  ex_train = NULL
                  ex_test = NULL

                } else {
                  ex_current = data[,best_params[1,'exogen'][[1]]]
                  ex_future = exogen_future[,best_params[1,'exogen'][[1]]]
                  ex_train = exogen_train[,best_params[1,'exogen'][[1]]]
                  ex_test = exogen_test[,best_params[1,'exogen'][[1]]]
                }
                
                # predict on test set one more time with best parameters for model accuracy info
                vc_train <- VECM(data.ts_train,
                                  r=r,
                                  lag=best_params[1,'lags'],
                                  include = as.character(best_params[1,'include']),
                                  estim = as.character(best_params[1,'estim']),
                                  exogen=ex_train)
                p <- as.data.frame(predict(vc_train,n.ahead=test_periods,exoPred=ex_test))
                p$xreg <- as.character(best_params[1,'exogen'])[[1]]
                p$model_form <- paste0('VECM with ',
                                        best_params[1,'lags'],' lags',
                                        '|estimator: ',best_params[1,'estim'],
                                        '|include: ',best_params[1,'include'],
                                        '|selected regressors: ',as.character(best_params[1,'exogen'])[[1]])

                write.csv(p,'tmp/tmp_test_results.csv',row.names=F)

                # train the final model on full dataset with best parameter values
                vc.out = VECM(data.ts,
                              r=r,
                              lag=best_params[1,'lags'],
                              include = as.character(best_params[1,'include']),
                              estim = as.character(best_params[1,'estim']),
                              exogen=ex_current
                )

                # make the forecast
                f <- as.data.frame(predict(vc.out,n.ahead=n_ahead,exoPred=ex_future))
                # write final forecast values
                write.csv(f,'tmp/tmp_forecast.csv',row.names=F)

                summary_df <- t(coef(vc_train))
                write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')

        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['test_set_predictions'] = tmp_test_results.iloc[:,0].to_list()
        self.info[call_me]['test_set_actuals'] = self.y[(-test_length):]
        self.info[call_me]['test_set_ape'] = [np.abs(y - yhat) / np.abs(y) for y, yhat in zip(self.y[(-test_length):],tmp_test_results.iloc[:,0])]
        self.info[call_me]['model_form'] = tmp_test_results['model_form'][0]
        self._metrics(call_me)
        self.forecasts[call_me] = tmp_forecast.iloc[:,0].to_list()
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

    def forecast_rf(self,test_length=1,Xvars='all',call_me='rf',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a random forest from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.ensemble import RandomForestRegressor
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = RandomForestRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Random Forest - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_gbt(self,test_length=1,Xvars='all',call_me='gbt',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a gradient boosting regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.ensemble import GradientBoostingRegressor
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = GradientBoostingRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Gradient Boosted Trees - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_adaboost(self,test_length=1,Xvars='all',call_me='adaboost',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with an ada boost regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.ensemble import AdaBoostRegressor
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = AdaBoostRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Ada Boosted Trees - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlp(self,test_length=1,Xvars='all',call_me='mlp',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a multi level perceptron neural network from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "mlp"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.neural_network import MLPRegressor
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = MLPRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Level Perceptron - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlr(self,test_length=1,Xvars='all',call_me='mlr',set_feature_importance=True):
        """ forecasts the stored y variable with a multi linear regression from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "mlr"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.linear_model import LinearRegression
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = LinearRegression()
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Linear Regression')
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_ridge(self,test_length=1,Xvars='all',call_me='ridge',alpha=1.0,set_feature_importance=True):
        """ forecasts the stored y variable with a ridge regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "ridge"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        alpha : float, default 1.0
                            the desired alpha hyperparameter to pass to the sklearn model
                            1.0 is also the default in sklearn
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.linear_model import Ridge
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = Ridge(alpha=alpha)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Ridge - {}'.format(alpha))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_lasso(self,test_length=1,Xvars='all',call_me='lasso',alpha=1.0,set_feature_importance=True):
        """ forecasts the stored y variable with a lasso regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "lasso"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        alpha : float, default 1.0
                            the desired alpha hyperparameter to pass to the sklearn model
                            1.0 is also the default in sklearn
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.linear_model import Lasso
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = Lasso(alpha=alpha)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Lasso - {}'.format(alpha))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_svr(self,test_length=1,Xvars='all',call_me='svr',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a support vector regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "mlp"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.svm import SVR
        self._ready_for_forecast(True)
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = self._get_info_dict()
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = SVR(**hyper_params)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Support Vector Regressor - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)


    def forecast_average(self,models='all',exclude=None,metric='mape',call_me='average',test_length='max'):
        """ averages a set of models to make a new estimator
            Parameters: models : list, "all", or starts with "top_", default "all"
                            "all" will average all models
                            starts with "top_" will average the top however many models are specified according to their respective metric values on the test set
                                the character after "top_" must be an integer
                                ex. "top_5"
                            if list, then those are the models that will be averaged
                        exclude : list, default None
                            manually exlcude some models
                            all models passed here will be excluded
                            if models parameters starts with "top" and one of those top models is in the list passed to exclude, that model will be excluded, and the other however many will be averaged (so you might only get 3 models averaged if you pass "top_4" for example)
                        metric : one of {'mape','rmse','mae','r2'}, default 'mape'
                        call_me : str, default "average"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        test_length : int or "max", default "max"
                            the test length to assign to the average model
                            if max, it will use the maximum test_length that all saved models can support
                            if int, will use that many test periods
                            if int is greater than one of the stored models' test length, this will fail
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        if models == 'all':
            avg_these_models = [e for e in list(getattr(self,metric).keys()) if (e != call_me) & (not e is None)]
        elif isinstance(models,list):
            avg_these_models = models[:]
        elif isinstance(models,str):
            if models.startswith('top_'):
                ordered_models = [e for e in self.order_all_forecasts_best_to_worst(metric) if (e != call_me) & (not e is None)]
                avg_these_models = [m for i, m in enumerate(ordered_models) if (i+1) <= int(models.split('_')[1])]
        else:
            raise ValueError(f'argument in models parameter not recognized: {models}')

        if not exclude is None:
            if not isinstance(exclude,list):
                raise TypeError(f'exclude must be a list type or None, not {type(exclude)} type')
            else:
                avg_these_models = [m for m in avg_these_models if m not in exclude]
            
        if len(avg_these_models) == 0:
            print('no models found to average')
            return None

        if test_length == 'max':
            for i, m in enumerate(avg_these_models):
                if i == 0:
                    test_length = self.info[m]['holdout_periods']
                else:
                    test_length = min(test_length,self.info[m]['holdout_periods'])
        else:
            assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
            assert test_length >= 1, 'test_length must be at least 1'

        self.mape[call_me] = 1
        self.forecasts[call_me] = [None]*self.forecast_out_periods

        self.info[call_me] = self._get_info_dict()
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['test_set_actuals'] = self.y[-(test_length):]

        forecasts = pd.DataFrame()
        test_set_predictions_df = pd.DataFrame()
        test_set_ape_df = pd.DataFrame()
        for m in avg_these_models:
            test_set_predictions_df[m] = self.info[m]['test_set_predictions'][-(test_length):]
            test_set_ape_df[m] = self.info[m]['test_set_ape'][-(test_length):] 
            forecasts[m] = self.forecasts[m]
            
        self.info[call_me]['model_form'] = 'Average of ' + str(len(avg_these_models)) + ' models: ' + ', '.join(avg_these_models)
        self.info[call_me]['test_set_predictions'] = list(test_set_predictions_df.mean(axis=1))
        self.info[call_me]['test_set_ape'] = list(test_set_ape_df.mean(axis=1))
        self._metrics(call_me)
        self.forecasts[call_me] = list(forecasts.mean(axis=1))

    def forecast_splice(self,models,periods,call_me='splice',**kwargs):
        """ splices multiple forecasts together
            this model will have no mape, test periods, etc, but will be saved in the forecasts attribute
            Parameters: models : list
                        periods : tuple of datetime objects
                            must be one less in length than models
                            each date represents a splice
                                model[0] --> :periods[0]
                                models[-1] --> periods[-1]:
                        call_me : str
                            the model nickname
                        key words should be the name of a metric ('mape','rmse','mae','r2') and a numeric value as the argument since some functions don't evaluate without numeric metrics
            >>> f.forecast_splice(models=['arima','tbats'],periods=(datetime.datetime(2020,1,1),))
        """
        assert isinstance(models,list), 'models must be a list'
        assert len(models) >= 2, 'need at least two models passed to models'
        assert np.array([m in self.forecasts.keys() for m in models]).all(), 'all models must have been evaluated already'
        assert isinstance(periods,tuple), 'periods must be a tuple'
        assert len(models) == len(periods) + 1, 'models must be exactly 1 greater in length than periods'
        assert np.array([p in self.future_dates for p in periods]).all(), 'all elements in periods must be datetime objects in future_dates'

        self.info[call_me] = self._get_info_dict()
        self.info[call_me]['model_form'] = "Splice of {}; splice point(s): {}".format(', '.join([k for k in self.info if k in models]), ', '.join([v.strftime('%Y-%m-%d') for v in periods]))
        self.forecasts[call_me] = [None]*self.forecast_out_periods

        for kw,v in kwargs.items():
            if kw in ('mape','rmse','mae','r2'):
                getattr(self,kw)[call_me] = v 
            else:
                raise ValueError(f'keyword {kw} not recognized!')
        
        # splice
        start = 0
        for i in range(len(periods)):
            end = [idx for idx,v in enumerate(self.future_dates) if v == periods[i]][0]
            self.forecasts[call_me][start:end] = self.forecasts[models[i]][start:end]
            start = end
        self.forecasts[call_me][start:] = self.forecasts[models[-1]][start:]

    def set_best_model(self,metric='mape'):
        """ sets the best forecast model based on which model has the best error metric value for the given holdout periods
            if two or more models tie, it will select whichever one was evaluated first
            Paramaters: metric : one of {'mape','rmse','mae','r2'}, default 'mape'
                            the error/accuracy metric to consider
        """
        self.best_model = self.order_all_forecasts_best_to_worst(metric)[0]

    def order_all_forecasts_best_to_worst(self,metric='mape'):
        """ returns a list of the evaluated models for the given series in order of best-to-worst according to the selected metric on the test set
            Paramaters: metric : one of {'mape','rmse','mae','r2'}, default 'mape'
                            the error/accuracy metric to consider
        """
        x = [h[0] for h in Counter(getattr(self,metric)).most_common()]
        if metric == 'r2':
            return x
        else:
            return x[::-1] # reversed copy of the list

    def pop(self,which):
        """ deletes a forecast or list of forecasts from the object
            Parameters: which : str or list-like
                if str, that model will be popped
                if not str, must be an iterable where elements are model nicknames stored in the object
            >>> f = Forecaster()
            >>> f.get_data_fred('UTUR')
            >>> f.forecast_auto_arima()
            >>> f.pop('auto_arima')
            >>> print(f.forecasts)
            {}
            >>> print(f.info)
            {}
            >>> print(f.mape)
            {}
        """
        if isinstance(which,str):
            self.forecasts.pop(which)
            self.mape.pop(which)
            self.info.pop(which)
            if which in self.feature_importance.keys():
                self.feature_importance.pop(which)

        else:
            for m in which:
                self.forecasts.pop(m)
                self.mape.pop(m)
                self.info.pop(m)
                if m in self.feature_importance.keys():
                    self.feature_importance.pop(m)

    def display_ts_plot(self,**kwargs):
        """ legacy method
        """
        self.plot(**kwargs)

    def plot(self,models='all',metric='mape',plot_fitted=False,print_model_form=False,print_metric=False):
        """ Plots time series results of the stored forecasts
            All models plotted in order of best-to-worst mapes
            Parameters: models : list, "all", or starts with "top_"; default "all"
                            the models you want plotted
                            if "all" plots all models
                            if list type, plots all models in the list
                            if starts with "top_" reads the next character(s) as the top however models you want plotted (based on lowest MAPE values)
                        plot_fitted : bool, default False
                            whether you want each model's fitted values plotted on the graph as a light dashed line
                            only works when graphing one model at a time (ignored otherwise)
                            this may not be available for some models
                        metric : one of {'mape','rmse','mae','r2'}
                            the error/accuracy metric to consider
                        print_model_form : bool, default False
                            whether to print the model form to the console of the models being plotted
                        print_metric : bool, default False
                            whether to print the metric specified in metric to the console of the models being plotted
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        if isinstance(models,str):
            if models == 'all':
                plot_these_models = self.order_all_forecasts_best_to_worst()[:]
            elif models.startswith('top_'):
                top = int(models.split('_')[1])
                if top > len(self.forecasts.keys()):
                    plot_these_models = self.order_all_forecasts_best_to_worst()[:]
                else:
                    plot_these_models = self.order_all_forecasts_best_to_worst()[:top]
            else:
                raise ValueError(f'models argument not supported: {models}')
        elif isinstance(models,list):
            plot_these_models = [m for m in self.order_all_forecasts_best_to_worst() if m in models]
        else:
            raise ValueError(f'models must be list or str, got {type(models)}')

        if (print_model_form) | (print_metric):
            for m in plot_these_models:
                print_text = '{} '.format(m)
                if print_model_form:
                    print_text += "model form: {} ".format(self.info[m]['model_form'])
                if print_metric:
                    print_text += "{}-period test-set {}: {} ".format(self.info[m]['holdout_periods'],metric,getattr(self,metric)[m])
                print(print_text)

        # plots with dates if dates are available, else plots with ambiguous integers
        sns.lineplot(x=self.current_dates,y=self.y)
        labels = ['Actual']

        for m in plot_these_models:
            # plots with dates if dates are available, else plots with ambiguous integers
            sns.lineplot(x=self.future_dates,y=self.forecasts[m])
            labels.append(f'{m} forecast')
            if plot_fitted & (not self.info[m]['fitted_values'] is None) & (len(plot_these_models) == 1):
                sns.lineplot(x=self.current_dates,y=self.info[m]['fitted_values'],style=True,dashes=[(2,2)],hue=.5)
                labels.append(f'{m} fitted values')

        plt.legend(labels=labels,loc='best')
        plt.xlabel('Date')
        plt.ylabel('{}'.format(self.name if not self.name is None else ''))
        plt.title(f'{self.name} Forecast Results')
        plt.show()

    def export_to_df(self,which='top_1',save_csv=False,metric='mape',csv_name='forecast_results.csv'):
        """ exports a forecast or forecasts to a pandas dataframe with future dates as the index and each exported forecast as a column
            returns a pandas dataframe
            will fail if you attempt to export forecasts of varying lengths
            by default, exports the best evaluated model by MAPE
            Parameters: which : starts with "top_", "all", or list; default "top_1"
                            which forecasts to export
                            if a list, should be a list of model nicknames
                        save_csv : bool, default False
                            whether to save the dataframe to a csv file in the current directory
                        metric : one of {'mape','rmse','mae','r2'}
                            the metric to use to evaluate best models
                        csv_name : str, default "forecat_results.csv"
                            the name of the csv file to be written out
                            ignored if save_csv is False
                            default pd.to_csv() called (comma delimited)
                            you can use this to change where the file is saved by specifying the file path
                                ex: csv_name = 'C:/NotWorkingDirectory/forecast_results.csv'
                                    csv_name = '../OtherParentDirectory/forecast_results.csv'
        """
        df = pd.DataFrame(index=self.future_dates)
        if isinstance(which,str):
            if (which == 'all') | (which.startswith('top_') & (int(which.split('_')[1]) > len(self.forecasts.keys()))):
                for m in self.order_all_forecasts_best_to_worst(metric)[:]:
                    df[m] = self.forecasts[m]
            elif which.startswith('top_'):
                top = int(which.split('_')[1])
                for m in self.order_all_forecasts_best_to_worst(metric)[:top]:
                    df[m] = self.forecasts[m]
            else:
                raise ValueError(f'which argument not supported: {which}')
        elif isinstance(which,list):
            for m in which[:]:
                df[m] = self.forecasts[m]
        else:
            raise ValueError(f'which argument not supported: {which}')

        if save_csv:
            df.to_csv(csv_name)

        return df