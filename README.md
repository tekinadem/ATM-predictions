


## USAGE
for Linear Model;
1. run `bin/validation.py`
2. run main-linear.sh (gitbash users)
   run main-linear.bat (windows cmd)
   run main-linear.ps1 (powershell users)

for SARIMAX Model;
1. run `bin/validation_sarimax.py`
2. run main-sarimax.sh (gitbash users)
   run main-sarimax.bat (windows cmd)
   run main-sarimax.ps1 (powershell users) 

for LSTM Model;
1. run `bin/validation_lstm.py`



 # Used/Validated Algorithms
  - a. Auto regressive models with exogenous variables
    - I. Using target variables' own lags in addition to variables such as weekday/month etc.
    - II. Linear models (OLS, Ridge)
    - III. Adding polynomial features
    - IV. Tree based models (decision tree, random forest regressor)
  - b. A neural network based sequence model
  - Used factor analysis because a weekly seasonality was identified (include box plot here)
  - Included lags of target variable because autocorrelation was identified. (acf and pacf plots)
  - linear model is considered because of linear relationship between lags.
  - tree based models were considered due to high number of categorical variables (e.g. day of week, day of month, weekday)

# Used Parameters
  - best number of lags were found based on acf and pacf plots
  - regularization parameter in ridge regression was found using a random search process
  - initial acf/pacf plots and residuals of sarimax were also used to determine AR, MA and seasonality parameters.
  - tree based model parameters were found using grid search CV
  - for LSTM, Adam optimization parameters were used 


