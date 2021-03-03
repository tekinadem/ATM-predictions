import pandas as pd
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from conf import (
    DATASET, DATASET_FILLED, LAGS, TEST_INDEX, TEXT_X, TEXT_Y, MODEL_NAME,
    FORECAST_PERIOD, FORECAST_BACKPERIOD, FORECAST_FILENAME, FIGURE_DIR)
from datetime import timedelta
from core.utils import preprocess
import joblib
import os


def main():
    df = pd.read_csv(DATASET_FILLED).assign(
        Date=lambda df: pd.to_datetime(df['Date'])).sort_values('Date')
    df_all = df.copy()

    if not FORECAST_BACKPERIOD:
        df = df.iloc[:].tail(LAGS + 1)
    else:
        df = df.iloc[:-FORECAST_BACKPERIOD].tail(LAGS + 1)

    df_orig = df.copy()
    model = joblib.load(MODEL_NAME)

    print('Using model:', model)

    forecasts = []

    for i in range(1, FORECAST_PERIOD + 1):
        df_orig = df_orig.append(pd.Series({'Date': df_orig['Date'].max() + timedelta(days=1), 'CashIn': None, 'CashOut': None}).fillna(0.), ignore_index=True)
        # import pdb; pdb.set_trace()
        dfX = preprocess(df_orig.copy(), LAGS).iloc[-1:, :]  # single row
        X = dfX.drop(['CashIn', 'CashOut'], 1)
        # import pdb; pdb.set_trace()
        # print(X)

        # y = dfX[['CashIn', 'CashOut']]

        lastday = df_orig['Date'].max()
        lastidx = df_orig.index.max()
        forecast = model.predict(X.iloc[-LAGS:])  # [[1234, 434677]]

        cashin = forecast[0, 0]
        cashout = forecast[0, 1]


        df_orig.loc[lastidx, 'CashIn'] = cashin
        df_orig.loc[lastidx, 'CashOut'] = cashout
        lastrow = pd.Series({'Date': lastday,
                             'CashIn': cashin, 'CashOut': cashout})
        forecasts.append(lastrow)
        # df_orig = df_orig.append(
        #     lastrow, ignore_index=True).iloc[-LAGS - 1:, :]

    fc = pd.concat(forecasts, axis=1)
    fc.index = ['Date', 'CashIn', 'CashOut']
    fc = fc.T
    fc = fc.set_index('Date')
    # fc.index = fc.index - timedelta(days=1)
    fc = fc.sort_index()
    print(fc)
    fc['CashOut'].plot(label='prediction')
    df_all.set_index('Date')[-TEST_INDEX:]['CashOut'].plot(label='actual')
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, 'predictions.png'))
    # plt.show()

    fc.iloc[1:].to_csv(FORECAST_FILENAME, index=True)


if __name__ == '__main__':
    main()
