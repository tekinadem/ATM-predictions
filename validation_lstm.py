import sys
sys.path.append('.')
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
from conf import DATASET, DATASET_FILLED, LAGS, TEST_INDEX, TEXT_X, TEXT_Y, MODEL_NAME, FIGURE_DIR, LSTM_EPOCHS, LSTM_NAME, LSTM_PATIENCE
from core.utils import mean_absolute_percentage_error, preprocess, preprocess_tsa, preprocess_lstm
from core.model import sarimax_conf, feature_ml_pipe
from core.lstm_model import lstm_model
import statsmodels.api as sm
import keras


def main():
    df = pd.read_csv(DATASET)
    X, y = preprocess_lstm(df, LAGS, feature_ml_pipe)
    y = y[['CashIn', 'CashOut']]

    X_train = X[:-TEST_INDEX]
    y_train = y[:-TEST_INDEX]
    X_test = X[-TEST_INDEX:]
    y_test = y[-TEST_INDEX:]

    model = lstm_model(input_shape=(LAGS, X.shape[2]), output_shape=2)
    model.compile('adam', loss='mae', metrics=['mse'])
    try:
        model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=LSTM_EPOCHS,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=LSTM_PATIENCE, monitor='val_loss', mode='min', restore_best_weights=True)
                ]
            )
    except KeyboardInterrupt:
        pass
    # y_pred = result.predict(start=y_test.index.min(), end=y_test.index.max(), exog=feature_ml_pipe.transform(X_test))

    asset = 'CashIn'
    y_pred = model.predict(X_test)[:, 0]
    y_pred = pd.Series(y_pred.ravel(), index=y_test.index).rename(asset)
    axnum = 0
    mse = metrics.mean_squared_error(y_test[asset], y_pred)
    mae = metrics.mean_absolute_error(y_test[asset], y_pred)
    r2 = metrics.r2_score(y_test[asset], y_pred)
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(y_test[asset], y_pred)

    print('Results for', asset)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    # print("Mean Absolute Percentage Error:", mape)
    print("R^2 Score:", r2)
    #
    fig, ax = plt.subplots(6, 1, figsize=(16, 12))
    fig.tight_layout()
    y_test[asset].plot(label='Actual', ax=ax[0])
    y_pred.plot(label='Predicted', ax=ax[0])
    ax[axnum].legend()
    ax[axnum].set_title(asset)
    #
    axnum += 1
    (y_pred - y_test[asset]).plot(ax=ax[axnum])
    ax[axnum].set_title(asset + " residuals")
    axnum += 1

    sm.graphics.tsa.plot_acf(y_pred - y_test[asset], ax=ax[axnum])
    ax[axnum].set_title(asset + " acf plot")


    asset = 'CashOut'
    y_pred = model.predict(X_test)[:, 1]
    y_pred = pd.Series(y_pred.ravel(), index=y_test.index).rename(asset)
    axnum += 1
    mse = metrics.mean_squared_error(y_test[asset], y_pred)
    mae = metrics.mean_absolute_error(y_test[asset], y_pred)
    r2 = metrics.r2_score(y_test[asset], y_pred)
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(y_test[asset], y_pred)
    #
    print('Results for', asset)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Percentage Error:", mape)
    print("R^2 Score:", r2)
    #
    # #
    y_test[asset].plot(label='Actual', ax=ax[axnum])
    y_pred.plot(label='Predicted', ax=ax[axnum])
    ax[axnum].legend()
    ax[axnum].set_title(asset)
    # ax[axnum].text(TEXT_X, TEXT_Y, f'Mean Absolute Error: {mae}')
    #
    axnum += 1

    (y_pred - y_test[asset]).plot(ax=ax[axnum])
    ax[axnum].set_title(asset + " residuals")
    #
    axnum += 1
    #
    sm.graphics.tsa.plot_acf(y_pred - y_test[asset], ax=ax[axnum])
    ax[axnum].set_title(asset + " acf plot")
    #
    fig.suptitle("Cash Predictions")
    # plt.show()
    fig.savefig(os.path.join(FIGURE_DIR, 'validation_figures_lstm.png'))
    #
    model.save(LSTM_NAME)
    print('Model saved to', LSTM_NAME)


if __name__ == '__main__':
    main()
