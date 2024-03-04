import lightgbm as lgb
from preprocessing import dataconnection
from preprocessing import reshape_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os


class LightGBM_Detection:

    def __init__(self):
        self.seed = 12345
        self.params = {
            'seed': self.seed,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

    def set_params(self, params):
        self.params = params

    def fit(self, dfX, dfZ, validation_size=0.2):

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

        X_training = lgb.Dataset(X_training_data, label=X_training_labels)
        X_test = lgb.Dataset(
            X_test_data, label=X_test_labels, reference=X_training)

        Z_training = lgb.Dataset(Z_training_data, Z_training_labels)
        Z_test = lgb.Dataset(
            Z_test_data, label=Z_test_labels, reference=Z_training)

        self.gbm_X = lgb.train(self.params, X_training,
                               valid_sets=[X_training, X_test])
        self.gbm_Z = lgb.train(self.params, Z_training,
                               valid_sets=[Z_training, Z_test])

        self.X_data = [X_training_data, X_test_data,
                       X_training_labels, X_test_labels]
        self.Z_data = [Z_training_data, Z_test_data,
                       Z_training_labels, Z_test_labels]

        self.full_X_predict_price = self.gbm_X.predict(self.feature_X)
        self.full_Z_predict_price = self.gbm_Z.predict(self.feature_Z)

    def predict(self, data, type):
        if type == 'X':
            return self.gbm_X.predict(data)
        elif type == 'Z':
            return self.gbm_Z.predict(data)
        else:
            raise RuntimeError("Unknown Medicine Type")

    import time
    time0 = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())

    def statistical_analysis(self, save=False, save_path='./RFD/lightgbm'+time0+'.xlsx'):
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
