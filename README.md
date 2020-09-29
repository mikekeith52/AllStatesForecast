# Productionalized Forecasting with Python
Builds the Forecaster object to create easily accessed forecasts for many time series

## Scenario
- Your company is wanting to expand into a new state
- As part of this decision, your role is to forecast the economic outlooks of all 50 states
- For each state, we will attempt to forecast a leading indicator, coincidental indicator, and unemployment rate for all states (150 total time series of varying lengths, each with its own peculiarities)
- All data obtained from [FRED](fred.stlouisfed.org)

## Other uses
- In the real world you may use this approach to predict metrics specific to a company, such as daily sales or monthly web interactions
- This framework can be expanded to such use cases with minimal adjustments--instead of modeling based on your beliefs about a recession, you may add a vector of holidays or recurring dates that you believe affect the metrics you are attempting to forecast
- In this forecast, there isn't much consideration of seasonality (two of the models used account for that automatically), but that is something you need to think about

## Installation (Windows Specific)
- In command line:
  - `cd path/to/local/directory`
  - `-r requirements.txt`

## rpy2
- Installing rpy2 can be tricky, you will need to have an R CRAN installed to PATH and add R_HOME and R_USER to your environment variables
  - R_USER (on my PC) : C:\Users\michaelkeith\Anaconda3\Lib\site-packages\rpy2
  - R_HOME (on my PC) : C:\Users\michaelkeith\Documents\R\R-4.0.2

## Output
![](https://github.com/mikekeith52/AllStatesForecast/blob/master/gif/gif.gif)