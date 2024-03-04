from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def Integration(DingDan, contat1, contat2, time0):
    inte = []
    con1 = contat1.values
    con2 = contat2.values

    for item in range(len(contat1)):
        temp = []

        for value in range(len(con1[item])):
            temp.append(con1[item, value])
        temp.append(con2[item])
        score = 0
        if temp[2] > 2:
            score = score+1
        if temp[4] > 2:
            score = score+1
        temp.append(score)
        inte.append(temp)

    inte = pd.DataFrame(inte)
    inte.columns = ['Id', 'leverage', 'SR_MR', 'cook', 'Res_RFD', 'score']

    pathway = './Output/'+time0+'.xls'
    wr = pd.ExcelWriter(pathway)
    inte.to_excel(wr, sheet_name='integration', index=False)
    wr.save()
    wr.close()  # 释放空间 Unleash the memory


def Outlier_RFD(name, time0):

    pred = pd.read_excel(io='./RFD/RFD'+time0+'.xlsx', sheet_name=name)
    Y = pred['Sold Price']
    Y_p = pred['Predicted Sold Price']
    ol = pred['Online Price']
    Y = np.array(Y/ol)
    Y_p = np.array(Y_p/ol)

    residuals = [Y, Y_p]
    res = np.std(residuals, axis=0)
    contat2 = pd.Series(res, name='Res_RFD')
    # studentized_residual 的绝对值不大于2
    # studentized_residual_out = X[np.abs(studentized_residuals) > 2]

    return contat2


def Outlier_Index_MR(X, Y, DingDan, model, time0):

    print(model.summary())

    Y_p = model.predict()

    print("RMSE : ", np.sqrt(mean_squared_error(Y, Y_p)))  # 计算 RMSE

    # 离群点检验
    out_points = model.get_influence()

    # 高杠杆值点（帽子矩阵）
    leverage = out_points.hat_matrix_diag
    # 高杠杆值点大于 2(p+1)/n时 被认为是异常点；其中p为维度，n为样本数量
    leverage_out = X[leverage > 2 * (X.shape[1]) / X.shape[0]]

    # DFFITS值
    # dffits = out_points.dffits[0]
    # DFFITS统计值大于 2sqrt((p+1)/n) 时被认为是异常点，其中p为维度，n为样本数量
    # diffts_out = X[dffits > 2 * np.sqrt((X.shape[1] + 1) / X.shape[0])]

    # 学生化残差
    residuals = Y-Y_p
    studentized_residuals = residuals / \
        np.sqrt(np.mean(residuals**2) * (1 - leverage))
    # studentized_residual 的绝对值不大于2
    studentized_residual_out = X[np.abs(studentized_residuals) > 2]

    # cook距离

    cook = out_points.cooks_distance[0]
    # 值的绝对值越大越有高概率是异常点

    # covratio值

    # covratio = out_points.cov_ratio
    # covratio值离 1 越远，越有可能是离群点

    # 将上面的几种异常值检验统计量与原始数据集合并
    contat1 = pd.concat([
        pd.Series(DingDan, name='id'),
        pd.Series(leverage, name='leverage'),
        # pd.Series(dffits, name='dffits'),
        pd.Series(studentized_residuals, name='rs'),
        pd.Series(cook, name='cook'),
        # pd.Series(covratio, name='covratio'),
    ], axis=1)

    return contat1