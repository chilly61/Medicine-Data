import numpy as np
import pandas as pd
import autoencoder as ae
import preprocessing
import multiregression as MR
from RF import RF_Detection
from LightGBM import LightGBM_Detection
# import HyperParameterTuning as tune
import Integration_output as Output
import time
time0 = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
start = time.perf_counter()
print('start')

# 输入文件路径
pathway = './reshaped_data.xls'
pathway_0 = './样例数据.xlsx'

# 得到预处理后的西药和中药dataframe以及挂网价格清单
onlineprice = preprocessing.reshape_data(pathway=pathway_0)

# 注意 真实数据集这里sheet_number应该为1
reshape_df = preprocessing.excelread(pathway, sheet_number=0)
df_X, df_Z = preprocessing.dataconnection(
    df_c=reshape_df, online_pirce_c=onlineprice)

print('preprocessing is okay!')

end1 = time.perf_counter()
runtime = end1-start
print("预处理时间：", runtime, "秒")

# 决策树部分
RFD = RF_Detection()
RFD.fit(df_X, df_Z)
scoreX = RFD.score('X')
scoreZ = RFD.score('Z')
print(scoreX, scoreZ)
RFD_result = RFD.statistical_analysis(save=True)

rf = pd.read_excel(io='./RFD/RFD.xlsx')
pathway = './RFD/RFD'+time0+'.xls'
wr = pd.ExcelWriter(pathway)
rf.to_excel(wr, sheet_name='Xiyao', index=False)
wr.save()
wr.close()  # 释放空间 Unleash the memory

print('RFD is okay!')

LGBM = LightGBM_Detection()
LGBM.fit(df_X, df_Z)
LGBM_result = LGBM.statistical_analysis(save=True)

rf = pd.read_excel(io='./RFD/lightgbm.xlsx')
pathway = './RFD/LGBM'+time0+'.xls'
wr = pd.ExcelWriter(pathway)
rf.to_excel(wr, sheet_name='Xiyao', index=False)
wr.save()
wr.close()  # 释放空间 Unleash the memory


print('LGBM is okay!')

end2 = time.perf_counter()
runtime = end2-end1
print("决策树运行时间：", runtime, "秒")

# 线性回归部分
Xi_x, Xi_y,  Xi_dingdan, Xi_model = MR.regression(df_X, 'X', time0)
# Zhong_x, Zhong_y, Zhong_dingdan, Zhong_model = MR.regression(df_Z, 'Z', time0)

print("MR is okay!")

end3 = time.perf_counter()
runtime = end3-end2
print("回归运行时间：", runtime, "秒")

# autoencoder部分
ae.encoder(onlineprice, df_X, df_Z, time0)
print('ae is okay!')

end4 = time.perf_counter()
runtime = end4-end3
print("autoencoder运行时间:", runtime, "秒")


con_x = Output.Outlier_Index_MR(Xi_x, Xi_y, Xi_dingdan, Xi_model, time0)
# con_y = Output.Outlier_Index_MR(
#     Zhong_x, Zhong_y, Zhong_dingdan, Zhong_model, time0)

con_21, con_22 = Output.Outlier_RFD('Xiyao', time0)

Output.Integration(Xi_dingdan, con_x, con_21, con_22, time0)


end5 = time.perf_counter()
runtime = end5-end4
print("integration运行时间:", runtime, "秒")

end = time.perf_counter()
runtime = end-start
print("总运行时间：", runtime, "秒")
