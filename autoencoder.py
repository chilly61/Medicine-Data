import pandas as pd
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split


def encoder(olprice, df_X, df_Z, time0):
    onlineprice = olprice

    df_Xiyao, df_Zhongyao = df_X, df_Z

    Original_Price = df_Xiyao[df_Xiyao['采购价格'] != 88888]

    Original_Price = Original_Price["采购价格"].values.astype(float)
    DingDan = df_Xiyao["订单代码"].values.astype(str)
    # 假设 df_Xiyao 已经是您从 dataconnection 函数中得到的 DataFrame

    # 删除“挂网价格”列
    df_Xiyao = df_Xiyao.drop("挂网价格", axis=1)
    df_Xiyao = df_Xiyao.drop('订单代码', axis=1)
    df_Xiyao = df_Xiyao[df_Xiyao['采购价格'] != 88888].copy()
    # 选择要进行onehot编码的列
    columns_to_onehot = ['药品标识码', '类别码', '名称码',
                         '剂型码', '规格包装码', '企业码', '地市', '医院等级', '订单时间']

    # 对指定列进行onehot编码，其他列保持原样
    onehot_encoded_df = pd.get_dummies(df_Xiyao[columns_to_onehot])

    df_Xiyao['Log_采购价格'] = np.log1p(df_Xiyao['采购价格'])
    df_not_onehot = df_Xiyao[['采购数量', 'Log_采购价格']].copy()  # 保留这些列，不进行onehot处理

    # 将未进行onehot编码的列添加回到onehot编码后的DataFrame中
    final_df = pd.concat([onehot_encoded_df, df_not_onehot], axis=1)

    # 现在 final_df 包含了onehot编码后的数据，以及未进行onehot编码的数字数据列

    input_data = final_df
    # print(input_data.shape[1])

    predict_price_index = input_data.columns.get_loc('Log_采购价格')  # 获取采购价格列的位置

    # 使用集成学习，训练五个autoencoder模型
    num_models = 5
    all_accuracies = []  # 存储所有模型的accuracy列表

    for i in range(num_models):
        # 数据集分割
        X_train, X_test = train_test_split(input_data, test_size=0.2)
        input_data = X_train
    input_size = 88
    hidden_size = 22
    code_size = 22
    output_size = 88

    x = Input(shape=(input_size,))

    # Encoder
    hidden_1 = Dense(hidden_size, activation='softplus')(x)
    h = Dense(code_size, activation='softplus')(hidden_1)

    # Decoder
    hidden_2 = Dense(hidden_size, activation='softplus')(h)
    r = Dense(output_size, activation='softplus')(hidden_2)

    autoencoder = Model(inputs=x, outputs=r)
    autoencoder.compile(optimizer='adam', loss='mse')

    # 套入autoencoder模型
    autoencoder.fit(input_data.values.astype(float), input_data.values.astype(
        float), batch_size=256, epochs=260, verbose=2)

    # 使用模型对输入数据进行预测
    reconstructed_data = autoencoder.predict(final_df.values.astype(float))

    # 将重构的数据转换为DataFrame
    df = pd.DataFrame(reconstructed_data)

    # 获取采购价格列的位置
    Predicted_Log_Price = reconstructed_data[:, predict_price_index]
    Predict_Price = np.expm1(Predicted_Log_Price)

    # 计算Accuracy
    Accuracy = [p / o if o != 0 else 0 for p,
                o in zip(Original_Price, Predict_Price)]
    all_accuracies.append(Accuracy)

    mean_accuracies = np.mean(all_accuracies, axis=0)
    # print(len(Predict_Price))
    # print(len(Original_Price))
    # print(len(mean_accuracies))

    # risk_scores = []
    # for accuracy in mean_accuracies:
    #     if accuracy > 1.15:
    #         risk_scores.append('high score')
    #     elif 1.10 <= accuracy <= 1.15:
    #         risk_scores.append('medium score')
    #     elif 1.05 <= accuracy < 1.10:
    #         risk_scores.append('low score')
    #     else:
    #         risk_scores.append('normal')
    # 计算重构误差
    squared_errors = (Predict_Price - Original_Price) ** 2

    # 计算平均值和标准差
    mean_squared_error = np.mean(squared_errors)
    std_squared_error = np.std(squared_errors)

    squared_differences = np.power(Predict_Price - Original_Price, 2)

    # 将阈值设置为平均值加上两倍标准差（正态分布）
    threshold1 = mean_squared_error + 2 * std_squared_error
    threshold2 = mean_squared_error + 3 * std_squared_error

    # 识别异常值：重构误差超过阈值的数据点
    outliers1 = squared_errors > threshold1
    outliers2 = squared_errors > threshold2

    # 合并到新的DataFrame
    result_df = pd.DataFrame({
        "Predict_Price": Predict_Price,
        "Original_Price": Original_Price,
        "Accuracy": mean_accuracies,
        "restructure_error": squared_errors
    })

    # 保存到Excel文件
    result_df.to_excel('/AE/'+time0+'result_data.xlsx', index=False, engine='openpyxl', columns=["Predict_Price",
                                                                                                 "Original_Price", "Accuracy",
                                                                                                 "restructure_error"])

    # 获取outlier1异常值的索引，去掉原始数据集中不必要的列，并写入Excel
    outlier1_indices = np.where(outliers1)[0]
    outlier1_data = df_Xiyao.iloc[outlier1_indices]
    outlier1_data = outlier1_data.drop("Log_采购价格", axis=1)
    outlier1_data.to_excel('outliers1_data.xlsx',
                           index=False, engine='openpyxl')

    # 获取outlier2异常值的索引，去掉原始数据集中不必要的列，并写入Excel
    outliers2_indices = np.where(outliers2)[0]
    outlier2_data = df_Xiyao.iloc[outliers2_indices]
    outlier2_data = outlier2_data.drop("Log_采购价格", axis=1)
    outlier2_data.to_excel('outliers2_data.xlsx',
                           index=False, engine='openpyxl')


# input_size = 88
# hidden_size = 22
# code_size = 22
# output_size = 88
#
# x = Input(shape=(input_size,))
#
# # Encoder
# hidden_1 = Dense(hidden_size, activation='softplus')(x)
# h = Dense(code_size, activation='softplus')(hidden_1)
#
# # Decoder
# hidden_2 = Dense(hidden_size, activation='softplus')(h)
# r = Dense(output_size, activation='softplus')(hidden_2)
#
# autoencoder = Model(inputs=x, outputs=r)
# autoencoder.compile(optimizer='adam', loss='mse')
#
# # 假设input_data是你的输入数据
# autoencoder.fit(input_data.values.astype(float), input_data.values.astype(float), batch_size=128, epochs=100, verbose=2)
#
# # 使用模型对输入数据进行预测
# reconstructed_data = autoencoder.predict(input_data.values.astype(float))
#
# # 将重构的数据转换为DataFrame
# df = pd.DataFrame(reconstructed_data)  # 根据你的数据适当调整列名
#
# # 获取采购价格列的位置
# Predicted_Log_Price = reconstructed_data[:, predict_price_index]
# Predict_Price = np.expm1(Predicted_Log_Price)
# # 计算Accuracy
# Accuracy = [p / o if o != 0 else 0 for p, o in zip(Original_Price, Predict_Price)]
# # print(len(Predict_Price))
# # print(len(Original_Price))
# # print(len(Accuracy))
#
# risk_scores = []
# for accuracy in Accuracy:
#     if accuracy > 1.15:
#         risk_scores.append('high score')
#     elif 1.10 <= accuracy <= 1.15:
#         risk_scores.append('medium score')
#     elif 1.05 <= accuracy < 1.10:
#         risk_scores.append('low score')
#     else:
#         risk_scores.append('normal')
#
# # 合并到新的DataFrame
# result_df = pd.DataFrame({
#     "Predict_Price": Predict_Price,
#     "Original_Price": Original_Price,
#     "Accuracy": Accuracy,
#     "risk_score": risk_scores
# })
#
# # 保存到Excel文件
# result_df.to_excel('result_data.xlsx', index=False, engine='openpyxl', columns=["Predict_Price",
#                                                                                 "Original_Price", "Accuracy",
#                                                                                 "risk_score"])
