import statsmodels.api as sm
import pandas as pd
import numpy as np


def ols_model(Y, X):
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # print(model.summary())

    return model


def regression(df_mr, type, time0):

    type_med = str(type)
    print(type_med)
    Z = df_mr["挂网价格"]
    DingDan = df_mr["订单代码"]
    onehot_df = df_mr

    index = []
    for line in range(len(onehot_df)):
        if onehot_df.loc[line, '采购价格'] == 88888:
            index.append(line)

    onehot_df = onehot_df.drop(index)
    onehot_df = onehot_df.drop('订单代码', axis=1)
    onehot_df['采购价格'] = onehot_df['采购价格']/onehot_df['挂网价格']
    onehot_df = pd.get_dummies(onehot_df)
    onehot_df = onehot_df.drop('挂网价格', axis=1)

    Z = Z.drop(index)
    DingDan = DingDan.drop(index)
    # X = onehot_df.drop("采购数量", axis=1)
    X = onehot_df.drop("采购价格", axis=1)
    if type_med == 'X':
        X = X.drop("药品标识码_X", axis=1)
    elif type_med == 'Z':
        X = X.drop("标识码_Z", axis=1)
    X = X.values
    Y = onehot_df["采购价格"]
    Y = Y.values

    Z = Z.values.astype(float)
    DingDan = DingDan.values.astype(str)
    model = ols_model(Y, X)
    Y_fitted = model.fittedvalues
    Y_0 = Y*Z
    Y_fitted_0 = Y_fitted*Z
    absolute = Y_0-Y_fitted_0
    proportions = absolute/Z
    bb = []
    for item in range(len(absolute)):
        aa = []
        aa.append(DingDan[item])
        aa.append(Y[item])
        aa.append(Y_fitted[item])
        aa.append(Y_0[item])
        aa.append(Y_fitted_0[item])
        aa.append(absolute[item])
        aa.append(proportions[item])
        aa.append(Z[item])
        bb.append(aa)

    bb = pd.DataFrame(bb)
    bb.columns = ['Id', 'Y_norm', 'Y_p_norm', 'Y', 'Y_p',
                  'diff', 'diff_proportion', 'online price']

    pathway = './MR/'+type_med+time0+'.xls'
    wr = pd.ExcelWriter(pathway)
    bb.to_excel(wr, sheet_name=type_med, index=False)
    wr.save()
    wr.close()  # 释放空间 Unleash the memory
    return X, Y, DingDan, model
