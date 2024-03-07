# 说明：
# df2array和array2df是数据格式转换函数，如果你使用过pandas，请忽略
# reshape_data为样例数据转换成较可观的随机价格的函数，请注意88888和99999两种极端情况，
# 99999代表原始数据挂网价格显示为空，88888代表原始数据无挂网价格，请在后续数据拟合过程中自行处理
# excelsave和excelread是数据存读函数，如果你接触过如何读写文件，请忽略
# dataconnection保留了原始数据集可用的部分，此处你可以进一步筛除无关或影响较小的因素
# 还请注意dataconnection返回的中药与西药数据集分别储存在不同df当中
import numpy as np
import pandas as pd
import xlrd
import random


def df2array(df):
    # dataframe的values默认格式为array，
    # 如果你想使用list，请使用tolist()将数组转为列表
    array = df.values
    return array


def array2df(arr, yourcolumn):
    # 请注意dataframe中列标签为column，行标签为index，
    # 如果你同时想要修改行标签，请在下面操作中加入index=yourindex
    dataframe = pd.DataFrame(arr, columns=yourcolumn)
    return dataframe


def reshape_data(pathway):
    # 读取第一个分文件，调用挂网价格和名称编码
    df_0 = pd.read_excel(io=pathway, sheet_name=0)
    # 读取原始数据集第二个分文件(交易)，read the second sheet in the raw file
    df_1 = pd.read_excel(io=pathway, sheet_name=1)

    price_dict = {}
    # 使用词典将第一个分文件中存在的编码和价格对应关系存储下来
    data = df_0.values
    for i in range(len(data)):
        if data[i][16] != '\\N':
            price_dict[data[i][4]] = data[i][16]
        else:
            price_dict[data[i][4]] = 99999

    return price_dict


def excelsave(pathway, save_df, sheet_name, your_label):
    wr = pd.ExcelWriter(pathway)
    save_df.to_excel(wr, sheet_name=sheet_name,
                     index=False, index_label=your_label)
    wr.save()
    wr.close()  # 释放空间 Unleash the memory


def excelread(pathway, sheet_number):  # 文件读取

    raw_dataframe = pd.read_excel(io=pathway, sheet_name=sheet_number)
    return raw_dataframe


def medstr2float(str1):
    line28 = int(0)
    if str1 == "无等级":
        line28 = 0
    elif str1 == "三级特等":
        line28 = 13
    elif str1 == "三级甲等":
        line28 = 12
    elif str1 == "三级乙等":
        line28 = 11
    elif str1 == "三级丙等":
        line28 = 10
    elif str1 == "三级无等":
        line28 = 9
    elif str1 == "二级甲等":
        line28 = 8
    elif str1 == "二级乙等":
        line28 = 7
    elif str1 == "二级丙等":
        line28 = 6
    elif str1 == "二级无等":
        line28 = 5
    elif str1 == "一级甲等":
        line28 = 4
    elif str1 == "一级乙等":
        line28 = 3
    elif str1 == "一级丙等":
        line28 = 2
    elif str1 == "一级无等":
        line28 = 1
    else:
        print("wrong!")
    return line28


def dataconnection(df_c, online_pirce_c):

    onlineprice = online_pirce_c
    # 使用词典将第一个分文件中存在的编码和价格对应关系存储下来
    data_c = df_c.values
    data_med = []

    df_X = pd.DataFrame(columns=['药品标识码', '类别码', '名称码', '剂型码',
                        '规格包装码', '企业码', '地市', '医院等级', '订单时间', '采购数量', '采购价格', '挂网价格', '订单代码'])
    df_Z = pd.DataFrame(
        columns=['标识码', '类别码', '名称码', '规格码', '企业码', '地市', '医院等级', '订单时间', '采购数量', '采购价格', '挂网价格', '订单代码'])

    order = []
    price = int(0)
    a = 0
    for line in data_c:
        if line[29] not in order:
            order.append(line[29])
            a = a+1
            med_code = line[4]
            if med_code in onlineprice:
                price = float(onlineprice[med_code])
            else:
                price = float(88888)
            line28 = medstr2float(line[28])
            if med_code[0] == 'X':
                code0 = med_code[0:1]
                temp_tuble = (med_code[0:1], med_code[1:6], med_code[6:10],
                              med_code[10:14], med_code[14:18], med_code[18:23],
                              line[2], line28, line[30].strftime('%Y-%m'), line[31], line[32], price, line[29])
                data_med.append(temp_tuble)
                df_X.loc[len(df_X)] = temp_tuble

            elif med_code[0] == 'Z':
                temp_tuble = (med_code[0:1], med_code[1:6],
                              med_code[6:11], med_code[11:15], med_code[15:20],
                              line[2], line28, line[30].strftime('%Y-%m'), line[31], line[32], price, line[29])
                data_med.append(temp_tuble)
                df_Z.loc[len(df_Z)] = temp_tuble

    return df_X, df_Z
