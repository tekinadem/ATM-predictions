import sys
sys.path.append('.')
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from conf import DATASET, DATASET_FILLED, LAGS, TEST_INDEX, TEXT_X, TEXT_Y, MODEL_NAME, FIGURE_DIR, MAIN_TEST_INDEX, SARIMAX_NAME
from core.utils import preprocess, mean_absolute_percentage_error
import joblib
import core.model as mdl
import core.utils as utils


def main():

    df = pd.read_csv(DATASET_FILLED)
    df = utils.get_dt_info(df)
    df = df.set_index('Date')
    train = df.loc[:]

    model = mdl.CascadeSARIMAX(mdl.WEEKLY_PARAMS_OUT, mdl.MONTHLY_PARAMS_OUT)
    model
    model.fit(train['CashOut'])
    model_in = mdl.CascadeSARIMAX(mdl.WEEKLY_PARAMS_IN, mdl.MONTHLY_PARAMS_IN)
    model_in
    model_in.fit(train['CashIn'])

    PREDICT_PARAMS = dict(start='2019-04-01', end='2019-04-30')

    pred_in = model_in.predict(**PREDICT_PARAMS).rename('CashIn')
    pred_out = model.predict(**PREDICT_PARAMS).rename('CashOut')

    predictions = pd.concat([pred_in, pred_out], axis=1)
    predictions.index.name = 'Date'
    predictions.to_csv('predictions_sarimax.csv', index=True)

    # Do not save the model due to 
    # joblib.dump(model, SARIMAX_NAME.format('out'))
    # joblib.dump(model_in, SARIMAX_NAME.format('in'))


if __name__ == '__main__':
    main()
