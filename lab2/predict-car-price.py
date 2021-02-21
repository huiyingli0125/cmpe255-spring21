import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)

        n = len(self.df)

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()

        y_train = df_train.msrp.values
        y_val = df_val.msrp.values
        y_test = df_test.msrp.values

        return df_train, df_val, df_test, y_train, y_val, y_test 

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
    
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        
        return w[0], w[1:]
    

    def prepare_X(self, df):
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df_num = df[base]
        print (df_num)
        df_num = df_num.fillna(0)
        X = df_num.values
        return X
    
    def predict(self, df, y):
        X = self.prepare_X(df)
        w_0, w = self.linear_regression(X, y)
        y_pred = w_0 + X.dot(w)
        return y_pred

    def rmse(self, y, y_pred) -> float:
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

def test() -> None:
    carPrice = CarPrice()
    carPrice.trim()
    df_train, df_val, df_test, y_train, y_val, y_test = carPrice.validate()
    y_train_pred = carPrice.predict(df_train, y_train)
    y_val_pred = carPrice.predict(df_val, y_val)
    y_test_pred = carPrice.predict(df_test, y_test)
    #train_rmse = carPrice.rmse(y_train, y_train_pred)
    #val_rmse = carPrice.rmse(y_val, y_val_pred)
    #test_rmse = carPrice.rmse(y_test, y_test_pred)
    df_test['msrp_pred'] = y_test_pred
    print (df_test.head(5))

if __name__ == "__main__":
    # execute only if run as a script
    test()