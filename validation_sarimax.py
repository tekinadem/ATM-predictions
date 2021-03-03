import sys
sys.path.append('.')

import os
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
import joblib
from conf import DATASET, DATASET_FILLED, LAGS, TEST_INDEX, TEXT_X, TEXT_Y, MODEL_NAME, FIGURE_DIR
from core.utils import mean_absolute_percentage_error, preprocess, preprocess_tsa
from core.model import ml_pipe, sarimax_conf, feature_ml_pipe
import seaborn as sns
import core.model as mdl
import core.utils as utils


def main():
    df = pd.read_csv(DATASET_FILLED)
    df = utils.get_dt_info(df)
    df = df.set_index('Date')
    train = df.loc[:'2019-02-28']
    test = df.loc['2019-03-01':]

    model = mdl.CascadeSARIMAX(mdl.WEEKLY_PARAMS_OUT, mdl.MONTHLY_PARAMS_OUT)
    model
    model.fit(train['CashOut'])
    model_in = mdl.CascadeSARIMAX(mdl.WEEKLY_PARAMS_IN, mdl.MONTHLY_PARAMS_IN)
    model_in
    model_in.fit(train['CashIn'])
    PREDICT_PARAMS = dict(start='2019-03-01', end='2019-03-31')
    fig, ax = plt.subplots(2, 1, figsize=(16,6))

    test['CashIn'].plot(label='actual', ax=ax[0])
    model_in.predict(start='2019-03-01', end='2019-03-31').plot(label='predicted', ax=ax[0])
    ax[0].set_title('Cash In Predictions')
    ax[0].legend()

    test['CashOut'].plot(label='actual', ax=ax[1])
    model.predict(start='2019-03-01', end='2019-03-31').plot(label='predicted', ax=ax[1])
    ax[1].set_title('Cash Out Predictions')
    fig.savefig(os.path.join(FIGURE_DIR, 'sarimax_validation.png'))
    # fig.show()
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    sm.graphics.tsa.plot_acf(model_in.resid, lags=120, ax=ax[0]);
    sm.graphics.tsa.plot_pacf(model_in.resid, lags=120, ax=ax[1]);
    fig.suptitle('ACF and PACF for Cash In Model');

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    sm.graphics.tsa.plot_acf(model.resid, lags=120, ax=ax[0]);
    sm.graphics.tsa.plot_pacf(model.resid, lags=120, ax=ax[1]);
    fig.suptitle('ACF and PACF for Cash Out Model');
    fig.savefig(os.path.join(FIGURE_DIR, 'sarimax_acf.png'))

    fig, ax = plt.subplots(2, 1, figsize=(16,8), sharex=True)
    sns.distplot(model.resid, ax=ax[0])
    ax[0].set_title('Distribution of residuals for Cash Out model')
    ax[0]

    sns.distplot(model_in.resid, ax=ax[1])
    ax[1].set_title('Distribution of residuals for Cash Out model')
    fig.savefig(os.path.join(FIGURE_DIR, 'sarimax_resid_dist.png'))

    pred_out = model.predict(**PREDICT_PARAMS)
    pred_in = model_in.predict(**PREDICT_PARAMS)

    mae_out = metrics.mean_absolute_error(test['CashOut'], pred_out)
    mae_in = metrics.mean_absolute_error(test['CashOut'], pred_in)
    r2_out = metrics.r2_score(test['CashOut'], pred_out)
    r2_in = metrics.r2_score(test['CashOut'], pred_in)
    mse_out = metrics.mean_squared_error(test['CashOut'], pred_out)
    mse_in = metrics.mean_squared_error(test['CashOut'], pred_in)
    rmse_out = mse_out**0.5
    rmse_in = mse_in**0.5


    print("MAE In:", mae_in)
    print("MSE In:", mse_in)
    print("RMSE In:", rmse_in)
    print("R2 In:", r2_in)

    print("MAE Out:", mae_out)
    print("MSE Out:", mse_out)
    print("RMSE Out:", rmse_out)
    print("R2 Out:", r2_out)

if __name__ == '__main__':
    main()
