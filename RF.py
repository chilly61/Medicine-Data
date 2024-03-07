import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


class RF_Detection:

    """Use random forest algorithm to conduct regression on the medical data, then detect outliers"""

    def __init__(self, random_seed=12345) -> None:
        self.seed = random_seed

    def fit(self, dfX, dfZ, validation_size=0.2, n_estimators=100, max_depth=None, max_features='sqrt'):

        df_X = dfX
        df_Z = dfZ

        df_X, self.online_price_X, self.order_num_X = self.__drop_invalid_data(
            df_X)
        df_Z, self.online_price_Z, self.order_num_Z = self.__drop_invalid_data(
            df_Z)

        self.feature_X, self.label_X = self.__get_feature_label(df_X)
        self.feature_Z, self.label_Z = self.__get_feature_label(df_Z)

        # split the training and test set (on Xiyao)
        X_training_data, X_test_data, X_training_labels, X_test_labels = train_test_split(
            self.feature_X, self.label_X, test_size=validation_size, random_state=self.seed)
        Z_training_data, Z_test_data, Z_training_labels, Z_test_labels = train_test_split(
            self.feature_Z, self.label_Z, test_size=validation_size, random_state=self.seed)

        self.X_data = [X_training_data, X_test_data,
                       X_training_labels, X_test_labels]
        self.Z_data = [Z_training_data, Z_test_data,
                       Z_training_labels, Z_test_labels]

        # use random forest to fit the training set
        self.X_regressor = ExtraTreesRegressor(n_estimators=n_estimators, random_state=self.seed,
                                               max_depth=max_depth, max_features=max_features).fit(X_training_data, X_training_labels)
        self.Z_regressor = ExtraTreesRegressor(n_estimators=n_estimators, random_state=self.seed,
                                               max_depth=max_depth, max_features=max_features).fit(Z_training_data, Z_training_labels)

        # perserve full regression prices for statistical analysis
        self.full_X_predict_price = self.X_regressor.predict(self.feature_X)
        self.full_Z_predict_price = self.Z_regressor.predict(self.feature_Z)

    def predict(self, data, type):
        if type == 'X':
            return self.X_regressor.predict(data)
        elif type == 'Z':
            return self.Z_regressor.predict(data)
        else:
            raise RuntimeError("Unknown Medicine Type")

    def statistical_analysis(self, save=False, save_path='./RFD/RFD.xlsx'):
        columns = ["Order Number", "Sold Price", "Predicted Sold Price",
                   "Online Price", "Sold/Online Proportion", "Predict/Online Proportion"]
        online_price_X = self.online_price_X.values.astype(float)
        result_X = np.column_stack((self.order_num_X.values, self.label_X,
                                    self.full_X_predict_price, online_price_X,
                                    self.label_X/online_price_X,
                                    self.full_X_predict_price/online_price_X))
        stats_X = pd.DataFrame(result_X, columns=columns)

        online_price_Z = self.online_price_Z.values.astype(float)
        result_Z = np.column_stack((self.order_num_Z.values,
                                    self.label_Z, self.full_Z_predict_price, online_price_Z,
                                    self.label_Z/online_price_Z, self.full_Z_predict_price/online_price_Z))
        stats_Z = pd.DataFrame(result_Z, columns=columns)
        predict_results = [stats_X, stats_Z]

        if save:
            self.save_result_csv(predict_results, save_path)
        return predict_results

    def save_result_csv(self, predict_results, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            predict_results[0].to_excel(
                writer, sheet_name='Xiyao', index=False)
            predict_results[1].to_excel(
                writer, sheet_name='Zhongyao', index=False)

    def score(self, type):
        if type == 'X':
            X_test_data = self.X_data[1]
            X_test_labels = self.X_data[3]
            return self.X_regressor.score(X_test_data, X_test_labels)
        elif type == 'Z':
            Z_test_data = self.Z_data[1]
            Z_test_labels = self.Z_data[3]
            return self.Z_regressor.score(Z_test_data, Z_test_labels)
        else:
            raise RuntimeError("Unknown Medicine Type")

    def test_regression_accuracy(self, Y_predict, Y_true):
        mae = mean_absolute_error(Y_true, Y_predict)
        mse = mean_squared_error(Y_true, Y_predict)
        rmse = np.sqrt(mse)
        return {'Mean Absolute Error': mae, 'Mean Squared Error': mse, 'Root Mean Squared Error': rmse}

    def __drop_invalid_data(self, df):
        unvalid_index = []
        for i in range(len(df)):
            if df.loc[i, "采购价格"] > 10000:
                unvalid_index.append(i)
        df = df.drop(unvalid_index)
        online_price_df = df["挂网价格"]
        order_num_df = df["订单代码"]
        df = df.drop(["挂网价格", "采购数量", "订单代码"], axis=1)
        return df, online_price_df, order_num_df

    def __get_feature_label(self, df):
        df_Y = df["采购价格"]
        df = df.drop("采购价格", axis=1)
        df = pd.get_dummies(df)
        return df.values, df_Y.values
