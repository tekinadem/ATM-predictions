import sys
sys.path.append('.')
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
import joblib
from conf import DATASET, DATASET_FILLED, LAGS, TEST_INDEX, TEXT_X, TEXT_Y, MODEL_NAME, FIGURE_DIR
from core.utils import mean_absolute_percentage_error, preprocess
from core.model import ml_pipe


def main():
    df = pd.read_csv(DATASET)
    dfX = preprocess(df, LAGS)
    dfX[['CashIn', 'CashOut']].to_csv(DATASET_FILLED, index=True)

    X = dfX.drop(['CashIn', 'CashOut'], 1)
    y = dfX[['CashIn', 'CashOut']]

    X_train = X.iloc[:-TEST_INDEX]
    y_train = y.iloc[:-TEST_INDEX]
    X_test = X.iloc[-TEST_INDEX:]
    y_test = y.iloc[-TEST_INDEX:]

    ml_pipe.fit(X_train, y_train)

    y_pred = ml_pipe.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

    asset = 'CashIn'
    axnum = 0
    mse = metrics.mean_squared_error(y_test[asset], y_pred[asset])
    mae = metrics.mean_absolute_error(y_test[asset], y_pred[asset])
    r2 = metrics.r2_score(y_test[asset], y_pred[asset])
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(y_test[asset], y_pred[asset])

    print('Results for', asset)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Percentage Error:", mape)
    print("R^2 Score:", r2)

    fig, ax = plt.subplots(6, 1, figsize=(16, 12))
    fig.tight_layout()
    # ax[axnum].plot(y_test[asset], label='Actual')
    # ax[axnum].plot(y_pred[asset], label='Predicted')
    y_test[asset].plot(label='Actual', ax=ax[0])
    y_pred[asset].plot(label='Predicted', ax=ax[0])
    ax[axnum].legend()
    ax[axnum].set_title(asset)
    ax[axnum].text(TEXT_X, TEXT_Y, f'Mean Absolute Error: {mae}')
    #
    axnum += 1

    (y_pred[asset] - y_test[asset]).plot(ax=ax[axnum])
    ax[axnum].set_title(asset + " residuals")

    axnum += 1

    sm.graphics.tsa.plot_acf(y_pred[asset] - y_test[asset], ax=ax[axnum])
    ax[axnum].set_title(asset + " acf plot")

    #
    asset = 'CashOut'
    axnum += 1
    mse = metrics.mean_squared_error(y_test[asset], y_pred[asset])
    mae = metrics.mean_absolute_error(y_test[asset], y_pred[asset])
    r2 = metrics.r2_score(y_test[asset], y_pred[asset])
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(y_test[asset], y_pred[asset])

    print('Results for', asset)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Percentage Error:", mape)
    print("R^2 Score:", r2)

    #
    y_test[asset].plot(label='Actual', ax=ax[axnum])
    y_pred[asset].plot(label='Predicted', ax=ax[axnum])
    ax[axnum].legend()
    ax[axnum].set_title(asset)
    ax[axnum].text(TEXT_X, TEXT_Y, f'Mean Absolute Error: {mae}')

    axnum += 1

    (y_pred[asset] - y_test[asset]).plot(ax=ax[axnum])
    ax[axnum].set_title(asset + " residuals")

    axnum += 1

    sm.graphics.tsa.plot_acf(y_pred[asset] - y_test[asset], ax=ax[axnum])
    ax[axnum].set_title(asset + " acf plot")

    fig.suptitle("Cash Predictions")
    # plt.show()
    fig.savefig(os.path.join(FIGURE_DIR, 'validation_figures.png'))


if __name__ == '__main__':
    main()
