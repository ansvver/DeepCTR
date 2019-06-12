from sklearn.model_selection import train_test_split
import pandas as pd
print('Load data...')
localPath='/data/code/douyinData/'
#localPath='D:/douyinData/'
df_train = pd.read_csv(localPath+'final_track2_train.txt',sep="\t",header=0,names=["uid","user_city","item_id","author_id","item_city","channel","finish","like","music_id","device","time","duration_time"])
print('查看全部数据')
print(df_train[:10])
trainData,valData=train_test_split(
            df_train,test_size=0.1,random_state=42)

print("训练集")
print(trainData[:10])
trainData.to_csv(localPath+"final_track2_forTrain.csv",sep="\t",header=False,encoding="utf-8")
valData.to_csv(localPath+"final_track2_forVal.csv",sep="\t",header=False,encoding="utf-8")