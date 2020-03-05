import pandas as pd
import numpy as np

files = ['keystroke_data-1.csv', 'keystroke_data-2.csv',
         'keystroke_data-3.csv', 'keystroke_data-4.csv', 'keystroke_data-5.csv']
df_list = []
for file_name in files:
    data = pd.read_csv('Data/' + file_name, header=None)

    new_data = pd.DataFrame()
    for i in range(0, data.shape[1], 2):
        new_data[str(i) + "H"] = (data.iloc[:, i+1] - data.iloc[:, i])
        if(i+2 < data.shape[1]):
            new_data[str(i) + 'DD'] = (data.iloc[:, i+2] - data.iloc[:, i])
            new_data[str(i) + 'UD'] = (data.iloc[:, i+2] - data.iloc[:, i+1])
    df_list.append(new_data)
    del new_data
result_1 = pd.concat(df_list)
result_1["Label"] = 0
files = [
    'time_BfBJvF.csv',
    'time_cgUZDy.csv',
    'time_chswjH.csv',
    'time_CjIwkR.csv',
    'time_cyCjXQ.csv',
    'time_DChksD.csv',
    'time_EyPbzF.csv',
    'time_fLcfEi.csv',
    'time_HyiMYA.csv',
    'time_mGASqx.csv',
    'time_qTHuas.csv',
    'time_UGoPId.csv',
    'time_uRVRut.csv',
    'time_wpVaeQ.csv',
    'time_XuPQkf.csv',
    'time_ysslXQ.csv',
    'time_zbuZes.csv'
]
df_list = []
counter = 1
for file_name in files:
    data = pd.read_csv('Data/' + file_name, header=None)

    new_data = pd.DataFrame()
    for i in range(0, data.shape[1], 2):
        new_data[str(i) + "H"] = (data.iloc[:, i+1] - data.iloc[:, i])
        if(i+2 < data.shape[1]):
            new_data[str(i) + 'DD'] = (data.iloc[:, i+2] - data.iloc[:, i])
            new_data[str(i) + 'UD'] = (data.iloc[:, i+2] - data.iloc[:, i+1])
    new_data["Label"] = counter
    counter += 1
    df_list.append(new_data)
    del new_data
result_2 = pd.concat(df_list)

result = pd.concat([result_1 , result_2])
result.to_csv("final_dataset.csv")
