# Forecaster User Guide

- Author Michael Keith: mikekeith52@gmail.com

[Overview](#overview)  
[Initializing Objects](#initializing-objects)  
[Setting Forecast Periods](#setting-forecast-periods)  
[Ingesting a DataFrame of External Regressors](#ingesting-a-dataframe-of-external-regressors)  
[Forecasting](#forecasting)  
[Plotting](#plotting)  
[Export Results](#export-results)  
[Examples](#examples)

## Overview:
- object to forecast time series data  
- natively supports the extraction of FRED data  
- the following models are supported:  
  - adaboost (sklearn)  
  - auto_arima (R forecast::auto.arima, no seasonal models)  
  - auto_arima_seas (R forecast::auto.arima, seasonal models)  
  - arima (statsmodels, not automatically optimized)
  - sarimax13 (Seasonal Auto Regressive Integrated Moving Average by X13 - R seasonal::seas)
  - average (any number of models can be averaged)
  - ets (exponental smoothing state space model - R forecast::ets)
  - gbt (gradient boosted trees - sklearn)
  - hwes (holt-winters exponential smoothing - statsmodels hwes)
  - auto_hwes (holt-winters exponential smoothing - statsmodels hwes)
  - lasso (sklearn)
  - mlp (multi level perceptron - sklearn)
  - mlr (multi linear regression - sklearn)
  - rf (random forest - sklearn)
  - ridge (sklearn)
  - svr (support vector regressor - sklearn)
  - tbats (exponential smoothing state space model With box-cox transformation, arma errors, trend, and seasonal component - R forecast::tbats)
  - nnetar (time series neural network - R forecast::nnetar)
  - var (vector auto regression - R vars::VAR)
  - vecm (vector error correction model - R tsDyn::VECM)
- for every evaluated model, the following information is stored in the object attributes:
  - in self.info (dict), a key is added as the model name and a nested dictionary as the value
    - the nested dictionary has the following keys:
      - 'holdout_periods' : int - the number of periods held out in the test set
      - 'model_form' : str - the name of the model with any hyperparameters, external regressors, etc
      - 'test_set_actuals' : list - the actual figures from the test set
      - 'test_set_predictions' : list - the predicted figures from the test set evaluated with a model from the training set
      - 'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
      - 'fitted_values' : list - the model's fitted values, when available. if not available, None
  - in self.mape (dict), a key is added as the model name and the Mean Absolute Percent Error as the value
  - in self.forecasts (dict), a key is added as the model name and a list of forecasted figures as the value
  - in self.feature_importance (dict), a key is added to the dictionary as the model name and the value is a dataframe that gives some info about the features' prediction power
    - if it is an sklearn model, it will be permutation feature importance from the eli5 package (https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)
    - any other model, it is a dataframe with at least the names of the variables in the index, with as much summary statistical info as possible
    - if the model doesn't use external regressors, no key is added here

## Initializing Objects

- there are two ways to initialize a forecaster object
  1. initialize empty and fill with FRED data:
```python
from Forecaster import Forecaster
f = Forecaster()
f.get_data_fred('UTUR')
print(f.y)
>>> [5.8, 5.8, ..., 5.0, 4.1]
print(f.current_dates)
>>> [Timestamp('1976-01-01 00:00:00'), Timestamp('1976-02-01 00:00:00'), ..., Timestamp('2020-09-01 00:00:00'), Timestamp('2020-10-01 00:00:00')]
print(f.name)
>>> 'UTUR'
```
  2. load data when initializing
```python
from Forecaster import Forecaster
f = Forecaster(y=[1,2,3,4,5],current_dates=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01']).to_list(),name='mydata')
```
- pay attention to the required types! if it says list, it does not mean list-like (yet)
- Parameters: 
  - **name** : str  
  - **y** : list  
  - **current_dates** : list  
    - an ordered list of dates that correspond to the ordered values in self.y  
    - elements must be able to be parsed by pandas as dates  
  - **future_dates** : list  
    - an ordered list of dates that correspond to the future periods being forecasted  
    - elements must be able to be parsed by pandas as dates  
  - **current_xreg** : dict  
  - **future_xreg** : dict  
  - **forecast_out_periods** : int, default length of future_dates or 24 if that is None  
  - all keyword arguments become attributes

## Setting Forecast Periods

1. `Forecaster.generate_future_dates(n,freq)`
- generates future dates and stores in the future_dates attribute
- changes forecast_out_periods attribute appropriately
- Parameters: 
  - **n** : int
    - the number of periods to forecast
    - the length of the resulting future_dates attribute
  - **freq** : str
    - a pandas datetime freq value
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
```python
f = Forecaster()
f.get_data_fred('UTUR')
f.generate_future_dates(12,'MS')
```
2. `Forecaster.set_forecast_out_periods(n)`
- sets the self.forecast_out_periods attribute and truncates self.future_dates and self.future_xreg if needed
- Parameters: 
  - **n** : int
    - the number of periods you want to forecast out for
    - if this is a larger value than the size of the future_dates attribute, some models may fail
    - if this is a smaller value the the size of the future_dates attribute, future_xreg and future_dates will be truncated
```python
f = Forecaster()
f.get_data_fred('UTUR')
f.generate_future_dates(12,'MS')
f.set_forecast_out_periods(6)
```

## Ingesting a DataFrame of External Regressors

- `Forecaster.process_xreg_df(self,xreg_df,date_col,process_columns=False)`
- takes a dataframe of external regressors
- any non-numeric data will be made into a 0/1 binary variable (using pandas.get_dummies(drop_first=True))
- deals with columns with missing data
- eliminates rows that don't correspond with self.y's timeframe
- splits values between future and current observations
- changes self.forecast_out_periods based on how many periods included in the dataframe past current_dates attribute
- assumes the dataframe is aggregated to the same timeframe as self.y (monthly, quarterly, etc.)
- for more complex processing, perform manipulations before passing through this function
- stores results in self.current_xreg, self.future_xreg, self.future_dates, and self.forecast_out_periods
- Parameters: 
  - **xreg_df** : pandas dataframe, required
    - this should include only independent regressors either in numeric form or that can be dummied into a 1/0 variable as well as a date column that can be parsed by pandas
    - do not include the dependent variable value
  - **date_col** : str, requried
    - the name of the date column in xreg_df that can be parsed with the pandas.to_datetime() function
  - **process_columns** : str, dict, or False; optional
    - how to process columns with missing data - most forecasts will not run when missing data is present in either xreg dict
    - supported: {'remove','impute_mean','impute_median','impute_mode','impute_min','impute_max',impute_0','forward_fill','backward_fill','impute_random'}
    - if str, must be one of supported and that method is applied to all columns with missing data
    - if dict, key is a column and value is one of supported, method only applied to columns with missing data                  
    - 'impute_random' will fill in missing values with random draws from the same column
```python           
xreg_df = pd.DataFrame({'date':['2020-01-01','2020-02-01','2020-03-01','2020-04-01']},'x1':[1,2,3,5],'x2':[1,3,3,3])
f = Forecaster(y=[4,5,9],current_dates=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01']).to_list())
f.process_xreg_df(xreg_df,date_col='date')
print(f.current_xreg)
>>> {'x1':[1,2,3],'x2':[1,3,3]}

print(f.future_xreg)
>>> {'x1':[5],'x2':[3]}

print(f.future_dates)
>>> [Timestamp('2020-04-01 00:00:00')]

print(f.forecast_out_periods)
>>> 1
```

## Forecasting
- methods: 
  - [forecast_auto_arima()](#forecast_auto_arima) 
  - [forecast_adaboost()](#forecast_adaboost) 
  - [forecast_auto_arima_seas()](#forecast_auto_arima_seas) 
  - [forecast_arima()](#forecast_arima)
  - [forecast_sarimax13()](#forecast_sarimax13)
  - [forecast_average()](#forecast_average)
  - [forecast_ets()](#forecast_ets)
  - [forecast_gbt()](#forecast_gbt)
  - [forecast_hwes()](#forecast_hwes)
  - [forecast_auto_hwes()](#forecast_auto_hwes)
  - [forecast_lasso()](#forecast_lasso)
  - [forecast_mlp()](#forecast_mlp)
  - [forecast_mlr()](#forecast_mlr)
  - [forecast_rf()](#forecast_rf)
  - [forecast_ridge()](#forecast_ridge)
  - [forecast_svr()](#forecast_svr)
  - [forecast_tbats()](#forecast_tbats)
  - [forecast_nnetar()](#forecast_nnetar)
  - [forecast_var()](#forecast_var)
  - [forecast_vecm()](#forecast_vecm)

### forecast_auto_arima
- `Forecaster.forecast_auto_arima(test_length=1,Xvars=None,call_me='auto_arima')`
- Auto-Regressive Integrated Moving Average   
- forecasts using auto.arima from the forecast package in R  
- uses an algorithm to find the best ARIMA model automatically by minimizing in-sample aic  
- does not search seasonal models  
- Parameters: 
  - **test_length** : int, default 1
      - the number of periods to holdout in order to test the model  
      - must be at least 1 (AssertionError raised if not)  
  - **Xvars** : list, "all", None, or starts with "top_", default None  
      - the independent variables used to make predictions  
      - if it is a list, will attempt to estimate a model with that list of Xvars  
      - if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars  
      - "top" is determined through absolute value of the pearson correlation coefficient on the training set  
      - if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available       - regressors that are not perfectly colinear and have variation  
      - if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation  
      - because the auto.arima function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is       - option  
      - if no arima model can be estimated, will raise an error
  - **call_me** : str, default "auto_arima"
      - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
```python
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
```

### forecast_auto_arima_seas
- `Forecaster.forecast_auto_arima_seas(start='auto',interval=12,test_length=1,Xvars=None,call_me='auto_arima_seas')`
- Auto-Regressive Integrated Moving Average   
- forecasts using auto.arima from the forecast package in R  
- searches seasonal models, but the algorithm isn't as complex as forecast_auto_arima() and is harder to set up  
- Parameters: 
  - **test_length** : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **start** : tuple of length 2 or "auto", default "auto"  
    - 1st element is the start year  
    - 2nd element is the start period in the appropriate interval  
    - for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)  
    - if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list   
  - **interval** : float, default 12  
    - the number of periods in one season (365.25 for annual, 12 for monthly, etc.)  
  - **Xvars** : list, "all", None, or starts with "top_", default None  
    - the independent variables used to make predictions  
    - if it is a list, will attempt to estimate a model with that list of Xvars  
    - if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars  
    - "top" is determined through absolute value of the pearson correlation coefficient on the training set  
    - if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation  
    - if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation  
    - because the auto.arima function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option  
    - if no arima model can be estimated, will raise an error  
  - **call_me** : str, default "auto_arima_seas"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_sarimax13
- `Forecaster.forecast_sarimax13(test_length=1,start='auto',interval=12,Xvars=None,call_me='sarimax13',error='raise')`
- Seasonal Auto-Regressive Integrated Moving Average - ARIMA-X13 - https://www.census.gov/srd/www/x13as/  
- Forecasts using the seas function from the seasonal package, also need the X13 software (x13as.exe) saved locally  
- Automatically takes the best model ARIMA model form that fulfills a certain set of criteria (low forecast error rate, high statistical significance, etc)  
- X13 is a sophisticated way to model seasonality with ARIMA maintained by the census bureau, and the seasonal package provides a simple wrapper around the software with R  
- The function here is simplified, but the power in X13 is its database offers precise ways to model seasonality, also takes into account outliers  
- Documentation: https://cran.r-project.org/web/packages/seasonal/seasonal.pdf, http://www.seasonal.website/examples.html  
- This package only allows for monthly or less granular observations, and only three years or fewer of predictions  
- the model will fail if there are 0 or negative values in the dependent variable attempted to be predicted  
- the model can fail for several other reasons (including lack of seasonality in the dependent variable)  
- Parameters: 
  - **test_length** : int, default 1  
      - the number of periods to holdout in order to test the model  
      - must be at least 1 (AssertionError raised if not)  
  - **start** : tuple of length 2 or "auto", default "auto"  
      - 1st element is the start year  
      - 2nd element is the start period in the appropriate interval  
      - for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)  
      - if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list  
  - **interval** : 1 of {1,2,4,12}, default 12  
      - 1 for annual, 2 for bi-annual, 4 for quarterly, 12 for monthly  
      - unfortunately, x13 does not allow for more granularity than the monthly level  
  - **Xvars** : list, "all", None, or starts with "top_", default None  
      - the independent variables used to make predictions  
      - if it is a list, will attempt to estimate a model with that list of Xvars  
      - if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars  
      - "top" is determined through absolute value of the pearson correlation coefficient on the training set  
      - if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation  
      - if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation  
      - because the seas function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option  
      - x13 already has an extensive list of x regressors that it will pull automatically--read the documentation for more info  
  - **call_me** : str, default "sarimax13"  
      - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
  - **error**: one of {"raise","pass","print"}, default "raise"  
      - if unable to estimate the model, "raise" will raise an error  
      - if unable to estimate the model, "pass" will silently skip the model and delete all associated attribute keys (self.info)  
      - if unable to estimate the model, "print" will skip the model, delete all associated attribute keys (self.info), and print the error  
      - errors are common even if you specify everything correctly -- it has to do with the X13 estimator itself  
      - one common error is caused when negative or 0 values are present in the dependent variables  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_arima
- `Forecaster.forecast_arima(test_length=1,Xvars=None,order=(0,0,0),seasonal_order=(0,0,0,0),trend=None,call_me='arima',**kwargs)`
- ARIMA model from statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- the args endog, exog, and dates passed automatically  
- the args order, seasonal_order, and trend should be specified in the method  
- all other arguments in the ARIMA() function can be passed to kwargs  
- using this framework, the following model types can be specified:  
  - AR, MA, ARMA, ARIMA, SARIMA, regression with ARIMA errors  
- this is meant for manual arima modeling; for a more automated implementation, see the forecast_auto_arima() and forecast_sarimax13() methods  
- Parameters: 
  - **test_length** : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **Xvars** : list, "all", or None default None  
    - the independent variables to use in the resulting X dataframes  
    - "top_" not supported  
  - **call_me** : str, default "arima"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
  - Info about all other arguments (order, seasonal_order, trend) can be found in the sm.tsa.arima.model.ARIMA documentation (linked above)  
  - other arguments from ARIMA() function can be passed as keywords  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults 

### forecast_hwes
- `Forecaster.forecast_hwes(test_length=1,call_me='hwes',**kwargs)`
- Holt-Winters Exponential Smoothing
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html  
- https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572  
- The Holt-Winters ES modifies the Holt ES technique so that it can be used in the presence of both trend and seasonality.  
- for a more automated holt-winters application, see forecast_auto_hwes()  
- if no keywords are added, this is almost always the same as a naive forecast that propogates the final value forward  
- Parameters: 
  - **test_length** : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **call_me** : str, default "hwes"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
  - keywords are passed to the ExponentialSmoothing function from statsmodels -- `dates` is specified automatically  
  - some important parameters to specify as key words: trend, damped_trend, seasonal, seasonal_periods, use_boxcox  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults   

### forecast_auto_hwes
- `Forecaster.forecast_auto_hwes(test_length=1,seasonal=False,seasonal_periods=None,call_me='auto_hwes')`
- Holt-Winters Exponential Smoothing  
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html  
- https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572  
- The Holt-Winters ES modifies the Holt ES technique so that it can be used in the presence of both trend and seasonality.  
- Will add different trend and seasonal components automatically and test which minimizes AIC of in-sample predictions  
- uses optimized model parameters to fit final model  
- for a more manual holt-winters application, see forecast_hwes()  
- Parameters: 
  - **test_length** : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **seasonal** : bool, default False  
    - whether there is seasonality in the series  
  - **seasonal_periods** : int, default None  
    - the number of periods to complete one seasonal period (for monthly, this is 12)  
    - ignored if seasonal is False  
  - **call_me** : str, default "hwes"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults   

### forecast_nnetar
- `Forecaster.forecast_nnetar(test_length=1,start='auto',interval=12,Xvars=None,P=1,boxcox=False,scale_inputs=True,repeats=20,negative_y='raise',call_me='nnetar')`
- Neural Network Time Series Forecast  
- uses nnetar function from the forecast package in R  
- this forecast does not work when there are negative or 0 values in the dependent variable  
- Parameters: 
  - **test_length** : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **start** : tuple of length 2 or "auto", default "auto"  
    - 1st element is the start year  
    - 2nd element is the start period in the appropriate interval  
    - for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)  
    - if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list  
  - **interval** : float, default 12  
    - the number of periods in one season (365.25 for annual, 12 for monthly, etc.)  
  - **Xvars** : list, "all", None, or starts with "top_", default None  
    - the independent variables used to make predictions  
    - if it is a list, will attempt to estimate a model with that list of Xvars  
    - if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars  
    - "top" is determined through absolute value of the pearson correlation coefficient on the training set  
    - if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation  
    - if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation  
    - because the function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option  
    - if no model can be estimated, will raise an error  
  - **P** : int, default 1  
    - the number of seasonal lags to add to the model  
  - **boxcox** : bool, default False  
    - whether to use a boxcox transformation on y  
  - **scale_inputs** : bool, default True  
    - whether to scale the inputs, performed after the boxcox transformation if that is set to True  
  - **repeats** : int, default 20  
    - the number of models to average with different starting points  
  - **negative_y** : one of {'raise','pass','print'}, default 'raise'    
    - what to do if negative or 0 values are observed in the y attribute   
    - 'raise' will raise a ValueError  
    - 'pass' will not attempt to evaluate a model without raising an error  
    - 'print' will not evaluate the model but print the error  
  - **call_me** : str, default "nnetar"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_tbats
- `Forecaster.forecast_tbats(test_length=1,season='NULL',call_me='tbats')`
- Exponential Smoothing State Space Model With Box-Cox Transformation, ARMA Errors, Trend And Seasonal Component  
- forecasts using tbats from the forecast package in R  
- auto-regressive only (no external regressors)  
- this is an automated model selection  
- Parameters: 
  - **test_length** : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **season** : int or "NULL"  
    - the number of seasonal periods to consider (12 for monthly, etc.)  
    - if no seasonality desired, leave "NULL" as this will be passed directly to the tbats function in r  
  - **call_me** : str, default "tbats"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_ets
- `Forecaster.forecast_ets(test_length=1,call_me='ets')`
- Exponential Smoothing State Space Model  
- forecasts using ets from the forecast package in R  
- auto-regressive only (no external regressors)  
- this is an automated model selection  
- Parameters: test_length : int, default 1  
    - the number of periods to holdout in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - call_me : str, default "ets"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_var
- `Forecaster.forecast_var(*series,auto_resize=False,test_length=1,Xvars=None,lag_ic='AIC',optimizer='AIC',season='NULL',max_externals=None,call_me='var')`
- Vector Auto Regression  
- forecasts using VAR from the vars package in R  
- Optimizes the final model with different time trends, constants, and x variables by minimizing the AIC or BIC in the training set  
- Unfortunately, only supports a level forecast, so to avoid stationarity issues, perform your own transformations before loading the data  
- Parameters: 
  - **series** : required  
    - lists of other series to run the VAR with  
    - each list must be the same size as self.y if auto_resize is False  
    - be sure to exclude NAs  
  - **auto_resize** : bool, default False  
    - if True, if series in series are different size than self.y, all series will be truncated to match the shortest series  
    - if True, note that the forecast will not necessarily make predictions based on the entire history available in y  
    - using this assumes that the shortest series ends at the same time the others do and there are no periods missing  
  - **test_length** : int, default 1  
    - the number of periods to hold out in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **Xvars** : list, "all", None, or starts with "top_", default None  
    - the independent variables used to make predictions  
    - if it is a list, will attempt to estimate a model with that list of Xvars  
    - if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars  
    - "top" is determined through absolute value of the pearson correlation coefficient on the training set  
    - if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors  
    - if it is "all", will attempt to estimate a model with all available x regressors  
    - because the VAR function will fail if there is perfect collinearity in any of the xregs or if there is no variation in any of the xregs, using "top_" is safest option  
  - **lag_ic** : str, one of {"AIC", "HQ", "SC", "FPE"}; default "AIC"  
    - the information criteria used to determine the optimal number of lags in the VAR function  
  - **optimizer** : str, one of {"AIC","BIC"}; default "AIC"  
    - the information criteria used to select the best model in the optimization grid  
    - a good, short resource to understand the difference: https://www.methodology.psu.edu/resources/AIC-vs-BIC/  
  - **season** : int, default "NULL"  
    - the number of periods to add a seasonal component to the model  
    - if "NULL", no seasonal component will be added  
    - don't use None ("NULL" is passed directly to the R CRAN mirror)  
    - example: if your data is monthly and you suspect seasonality, you would make this 12  
  - **max_externals** : int or None type, default None  
    - the maximum number of externals to try in each model iteration  
    - 0 to this value of externals will be attempted and every combination of externals will be tried  
    - None signifies that all combinations will be tried  
    - reducing this from None can speed up processing and reduce overfitting  
  - **call_me** : str, default "var"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_vecm  
- `Forecaster.forecast_vecm(*cids,auto_resize=False,test_length=1,Xvars=None,r=1,max_lags=6,optimizer='AIC',max_externals=None,call_me='vecm')`
- Vector Error Correction Model
- forecasts using VECM from the tsDyn package in R
- Optimizes the final model with different lags, time trends, constants, and x variables by minimizing the AIC or BIC in the training set
- Parameters: 
  - **cids** : required  
    - lists of cointegrated data  
    - each list must be the same size as self.y  
    - if this is only 1 list, it must be cointegrated with self.y  
    - if more than 1 list, there must be at least 1 cointegrated pair between cids* and self.y (to fulfill the requirements of VECM)  
    - be sure to exclude NAs  
  - **auto_resize** : bool, default False  
    - if True, if series in cids are different size than self.y, all series will be truncated to match the shortest series  
    - if True, note that the forecast will not necessarily make predictions based on the entire history available in y  
    - using this assumes that the shortest series ends at the same time the others do and there are no periods missing  
  - **test_length** : int, default 1  
    - the number of periods to hold out in order to test the model  
    - must be at least 1 (AssertionError raised if not)  
  - **Xvars** : list, "all", None, or starts with "top_", default None  
    - the independent variables used to make predictions  
    - if it is a list, will attempt to estimate a model with that list of Xvars  
    - if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars  
    - "top" is determined through absolute value of the pearson correlation coefficient on the training set  
    - if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors  
    - if it is "all", will attempt to estimate a model with all available x regressors  
    - because the VECM function will fail if there is perfect collinearity in any of the xregs or if there is no variation in any of the xregs, using "top_" is safest option  
  - **r** : int, default 1  
    - the number of total cointegrated relationships between self.y and cids  
    - if not an int or less than 1, an AssertionError is raised  
  - **max_lags** : int, default 6  
    - the total number of lags that will be used in the optimization process  
    - 1 to this number will be attempted  
    - if not an int or less than 0, an AssertionError is raised  
  - **optimizer** : str, one of {"AIC","BIC"}; default "AIC"  
    - the information criteria used to select the best model in the optimization grid  
    - a good, short resource to understand the difference: https://www.methodology.psu.edu/resources/AIC-vs-BIC/  
  - **max_externals** : int or None type, default None  
    - the maximum number of externals to try in each model iteration  
    - 0 to this value of externals will be attempted and every combination of externals will be tried  
    - None signifies that all combinations will be tried  
    - reducing this from None can speed up processing and reduce overfitting  
  - **call_me** : str, default "vecm"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

### forecast_average
- `Forecaster.forecast_average(models='all',exclude=None,call_me='average',test_length='max')`
- averages a set of models to make a new estimator  
- Parameters: 
  - **models** : list, "all", or starts with "top_", default "all"  
    - "all" will average all models  
    - starts with "top_" will average the top however many models are specified according to their respective MAPE values on the test set (lower = better)  
      - the character after "top_" must be an integer  
      - ex. "top_5"  
    - if list, then those are the models that will be averaged  
  - **exclude** : list, default None  
    - manually exlcude some models  
    - all models passed here will be excluded  
    - if models parameters starts with "top" and one of those top models is in the list passed to exclude, that model will be excluded, and the other however many will be averaged (so you might only get 3 models averaged if you pass "top_4" for example)  
  - **call_me** : str, default "average"  
    - the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries  
  - **test_length** : int or "max", default "max"  
    - the test length to assign to the average model  
    - if max, it will use the maximum test_length that all saved models can support  
    - if int, will use that many test periods  
    - if int is greater than one of the stored models' test length, this will fail  
- See [forecast_auto_arima()](#forecast_auto_arima) documentation for an example of how to call a forecast method and access reults  

## Plotting
- `Forecaster.plot(models='all',plot_fitted=False,print_model_form=False,print_mapes=False)`
- Plots time series results of the stored forecasts  
- All models plotted in order of best-to-worst mapes  
- Parameters: 
  - **models** : list, "all", or starts with "top_"; default "all"  
    - the models you want plotted  
    - if "all" plots all models  
    - if list type, plots all models in the list  
    - if starts with "top_" reads the next character(s) as the top however models you want plotted (based on lowest MAPE values)  
  - **plot_fitted** : bool, default False  
    - whether you want each model's fitted values plotted on the graph as a light dashed line  
    - only works when graphing one model at a time (ignored otherwise)  
    - this may not be available for some models  
  - **print_model_form** : bool, default False  
    - whether to print the model form to the console of the models being plotted  
  - **print_mapes** : bool, default False  
    - whether to print the MAPEs to the console of the models being plotted  
```python
f.plot()
f.plot(models=['arima','tbats'])
f.plot(models='top_4')
f.plot(models='top_1',print_mapes=True,plot_fitted=True)
```

## Examples

[Analysis 1](#analysis-1): Forecasting statewide indicators, using automatic adjustments for stationarity  
[Analysis 2](#analysis-2): Forecasting statewide indicators, taking differences to account for stationarity  
[Analysis 3](#analysis-3): Forecasting statewide indicators, using a dataframe of external regressors  
[Analysis 4](#analysis-4): Forecasting different length series  
[Analysis 5](#analysis-5): three ways to auto-forecast seasonality  
[Analysis 6](#analysis-6): using the same model with different parameters  
[Analysis 7](#analysis-7): forecasting with a vecm  

### Analysis 1: Forecasting statewide indicators, using automatic adjustments for stationarity
```python
from Forecaster import Forecaster

state_forecasts = {}
for state in states: # states is a list of state abbreviations
  for ei in ('PHCI','UR','SLIND'):
    f = Forecaster()
    f.get_data_fred(state+ei)
    f.generate_future_dates(12,'MS') # generate 12 monthly future dates to forecast 12 months to the future
    f.forecast_auto_arima(test_length=12)
    f.forecast_auto_hwes(test_length=12)
    f.forecast_average()
    state_forecasts[state+ei] = f

# compare results
state = 'UT'
ei = 'UR'
forecast = state_forecasts[state+ei]
print(forecast.mape)
print(forecast.info)
forecast.plot()
```
### Analysis 2: Forecasting statewide indicators, taking differences to account for stationarity
```python
from Forecaster import Forecaster

state_forecasts = {}
for state in states: # states is a list of state abbreviations
  for ei in ('PHCI','UR','SLIND'):
    f = Forecaster()
    if ei == 'PHCI':
      f.get_data_fred(state+ei,i=2) # two differences to make this one stationary
    else:
      f.get_data_fred(state+ei,i=1) # one difference for these ones

    f.generate_future_dates(12,'MS') # generate 12 monthly future dates to forecast 12 months to the future
    f.forecast_ets(test_length=12)
    f.forecast_tbats(test_length=12)
    f.forecast_average()
    state_forecasts[state+ei] = f

# compare results
state = 'UT'
ei = 'UR'
forecast = state_forecasts[state+ei]
print(forecast.mape)
print(forecast.info)
forecast.plot()
```
### Analysis 3: Forecasting statewide indicators, using a dataframe of external regressors
```python
from Forecaster import Forecaster
df = pd.read_csv('path/to/external/regressors.csv')

state_forecasts = {}
for state in states: # states is a list of state abbreviations
  for ei in ('PHCI','UR','SLIND'):
    f = Forecaster()
    f.get_data_fred(state+ei)
    f.process_xreg_df(df,date_col='Date',impute_missing='backward_fill') # automatically sets future dates and forecast periods
    f.forecast_mlr(test_length=12)
    f.forecast_svr(test_length=12)
    f.forecast_mlp(test_length=12)
    f.forecast_rf(test_length=12)
    f.forecast_average(models='top_2')
    state_forecasts[state+ei] = f

# compare results
state = 'UT'
ei = 'UR'
forecast = state_forecasts[state+ei]
print(forecast.mape)
print(forecast.info)
forecast.plot()

```
### Analysis 4: Forecasting different length series
```python
from Forecaster import Forecaster

df = pd.read_csv('path/to/df/with/time/series.csv',index_col=0) # each column is a series to forecast and the index is a datetime, daily data
externals = pd.read_csv('path/to/externals.csv') # date should be a column, not index


for c in df.columns:
  y_load = df[c].drop_na() # nas are present when a certain series doesn't have data starting at the index's beginning
  f = Forecaster(y=y_load.to_list(),current_dates=y_load.index.to_list(),name=c)
  f.process_xreg_df(externals,date_col='Date')
  test_length = 5 if len(y_load) < 100 else 30: # 5 day test length if less than 100 days of data, otherwise 30 days
  if test_length == 5: # for shorter data series, you can use models that work better with shorter series
    f.forecast_ets(test_length=test_length)
    f.forecast_tbats(test_length=test_length)
    f.forecast_auto_hwes(test_length=test_length)
  else: # use more advanced techniques that utilize the regressors
    for i in range(3): f.forecast_auto_arima(test_length=test_length,Xreg=f'top_{i}',call_me=f'arima_{i}_reg') # 0, 1, 2 regressors
    f.forecast_nnetar(test_length=test_length,Xreg=['x1','x2','x4'],P=0)
    f.forecast_arima(test_length=test_length,order=(1,1,1),Xreg='all',call_me='arima_all_reg')
  f.forecast_average(models='top_5')
```

### Analysis 5: three ways to auto-forecast seasonality
```python
from Forecaster import Forecaster
def main():
  f = Forecaster()
  f.get_data_fred('HOUSTNSA')

  f.generate_future_dates(12,'MS')

  f.forecast_auto_hwes(test_length=12,seasonal=True,seasonal_periods=12)

  print(f.mape['auto_hwes'])

  f.forecast_auto_arima_seas(test_length=12)

  print(f.info['auto_arima_seas']['model_form'])
  print(f.mape['auto_arima_seas'])

  f.forecast_sarimax13(test_length=12,error='ignore')

  print(f.info['sarimax13']['model_form'])
  print(f.feature_importance['sarimax13'].index.to_list())
  print(f.mape['sarimax13'])

  f.forecast_average(models=['auto_arima_seas','sarimax13'])

  print(f.mape['average'])

  f.plot()

if __name__ == '__main__':
  main()
```

### Analysis 6: using the same model with different parameters
```python
from Forecaster import Forecaster

df = pd.read_csv('path/to/df/with/time/series.csv',index_col=0) # each column is a series to forecast and the index is a datetime
externals = pd.read_csv('path/to/externals.csv') # date should be a column, not index

for c in df.columns:
  y_load = df[c]
  f = Forecaster(y=y_load.to_list(),current_dates=y_load.index.to_list(),name=c)
  f.process_xreg_df(externals,date_col='Date')
  f.forecast_nnetar(test_length=12,Xvars=None,P=0,boxcox=True,call_me='nnetar_1') # no seasonality, no xregs, use boxcox
  f.forecast_nnetar(test_length=12,Xvars='all',P=0,boxcox=False,call_me='nnetar_2') # no seasonality, all xregs, no boxcox
  f.forecast_nnetar(test_length=12,Xvars='all',P=0,boxcox=False,repeats=10,call_me='nnetar_3') # no seasonality, all xregs, 10 repeates, no boxcox
  f.forecast_nnetar(test_length=12,Xvars='top_3',P=1,boxcox=False,call_me='nnetar_4') # 1 difference seasonality, top 3 xregs, no boxcox
  f.forecast_average(models=['nnetar_1','nnetar_3'],call_me='combo_1') # combo of two neetar models
  f.forecast_average(models=['nnetar_2','nnetar_4'],call_me='combo_2') # combo of two neetar models
  f.forecast_average(models='top_2',call_me='combo_3') # combo of top_2 best models
  f.forecast_average(models=['combo_1','combo_2','combo_3'],call_me='combo_combo') # combo of combos
  f.forecast_average(models='top_3',exclude=[m for m in f.forecasts.keys() if m.startswith('combo')]) # top_3 nnetars combo
```

### Analysis 7: forecasting with a vecm  
```python
from Forecaster import Forecaster

df_male = pd.read_csv('path/to/df/with/time/series_male.csv',index_col=0) # each column is a series to forecast and the index is a datetime
df_female = pd.read_csv('path/to/df/with/time/series_female.csv',index_col=0) # each column is a series to forecast and the index is a datetime
externals = pd.read_csv('path/to/externals.csv') # date should be a column, not index

# we know male and female series are cointegrated

# you can make a function to save all info from one forecasted VECM into another object
def save_info_about_other_series(Forecaster_object_1,Forecaster_object_2,pos=1,call_me='vecm'):
  """ save all info from one to the other, but it should be called right after you run one vecm so that no R tmp data is overwritten
      pos is what position (0-indexed) to get data from the tmp R data
  """
  fo1, fo2 = Forecaster_object_1, Forecaster_object_2
  fo2.mape[call_me] = fo1.mape[call_me]
  fo2.info[call_me] = fo1.info[call_me]
  fo2.info[call_me]['actuals'] = pd.read_csv('tmp/tmp_test_results.csv').iloc[:,pos].to_list()
  fo2.feature_importance[call_me] = fo1.feature_importance[call_me]
  fo2.forecasts[call_me] = pd.read_csv('tmp/tmp_forecast.csv').iloc[:,pos].to_list()

forecasts = {}
for c in df_male.columns:
  y_load_male = df_male[c]
  f_male = Forecaster(y=y_load_male.to_list(),current_dates=y_load_male.index.to_list(),name=c)
  f_male.process_xreg_df(externals,date_col='Date')
  y_load_female = df_female[c]
  y_female = Forecaster(y=y_load_female.to_list(),current_dates=y_load_female.index.to_list(),name=c)
  f_male.forecast_vecm(y_female.y,auto_resize=True,test_length=12,r=1,Xvars='top_5',max_lags=12,optimizer='BIC',max_externals=3)
  save_info_about_other_series(f_male,f_female,1,'vecm')
```