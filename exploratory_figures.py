import matplotlib.pyplot as plt
import pandas as pd
from conf import DATASET, FIGURE_DIR, DATASET_FILLED
import os
import statsmodels.api as sm

def main():
    df = pd.read_csv(DATASET_FILLED)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df['CashIn'].hist()
    plt.savefig(os.path.join(FIGURE_DIR, 'CashIn_hist.png'))
    plt.close()
    z = sm.tsa.seasonal_decompose(df['CashIn'])
    z.plot()
    plt.show()
    plt.close()
    sm.graphics.tsa.plot_acf(z.resid.fillna(0))
    plt.show()
    plt.close()
    sm.graphics.tsa.plot_pacf(z.resid.fillna(0))
    plt.show()
    plt.close()
    sm.graphics.tsa.plot_pacf(df['CashIn'] - z.seasonal)
    sm.graphics.tsa.plot_acf(df['CashIn'] - z.seasonal)
    plt.show()
    plt.close()
    print(z.seasonal)


if __name__ == '__main__':
    main()
