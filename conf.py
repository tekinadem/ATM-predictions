"""Project configuration."""

DATASET = "data/ds_exercise_data.csv"
DATASET_FILLED = "data/ds_exercise_filled.csv"
FORECAST_FILENAME = 'predictions.csv'
LAGS = 18
TEST_INDEX = 31
MAIN_TEST_INDEX = 0
FORECAST_BACKPERIOD = 1  # set this to 0 for actual forecast
FORECAST_PERIOD = 31
TEXT_Y = 80000
TEXT_X = '2016-07-01'
MODEL_NAME = 'models/linear.joblib'
FIGURE_DIR = 'fig'
LSTM_EPOCHS = 9000
LSTM_NAME = 'models/AdemNet-ATM'
LSTM_PATIENCE = 200
SARIMAX_NAME = 'models/sarimax_{}.joblib'
