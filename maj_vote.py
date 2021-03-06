import pandas as pd
import numpy as np
from scipy.stats import mode
df1 = pd.read_csv("pred_bert_mix_simple_cleanup_ep2.csv").to_numpy()
df2 = pd.read_csv("pred_mix_bert_fullds_ep1.csv").to_numpy()
df3 = pd.read_csv("pred_mix_bert_multilingual.csv").to_numpy()
df4 = pd.read_csv("pred_mix_bert_weight_decay.csv").to_numpy()
df5 = pd.read_csv("pred_roberta_mix_ep4_s2.csv").to_numpy()
df6 = pd.read_csv("pred_roberta_share6_ep3_s0.csv").to_numpy()
df7 = pd.read_csv("pred_roberta_ws4_ep3_s0.csv").to_numpy()

res = np.empty((df1.shape[0],7))
res[:,1] = df2[:,1]
res[:,0] = df1[:,1]
res[:,2] = df3[:,1]
res[:,3] = df4[:,1]
res[:,4] = df5[:,1]
res[:,5] = df6[:,1]
res[:,6] = df7[:,1]

print(res)
df,_ = mode(res,axis=1)
print(df)
df = pd.DataFrame({'Prediction': df[:, 0]})
print(df)

#df.rename(columns={ df.columns[0]: "Id",df.columns[1]: "Prediction" }, inplace = True)
df.index = df.index + 1
#df.index="Id"

#df *= -1
df.astype('int32')
print(df)
print(df['Prediction'].value_counts())

df.to_csv('onlyroberta.csv',float_format='%.f')
df = pd.read_csv('onlyroberta.csv')
df.rename(columns={ df.columns[0]: "Id",df.columns[1]: "Prediction" }, inplace = True)
df.to_csv('onlyroberta.csv',index = False)




