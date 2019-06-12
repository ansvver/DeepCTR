import  configparser
import os
import sys
from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from collections import Counter
import datetime
from pyspark.sql.functions import udf
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import gc

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

pathFile="D:/douyinData/test.txt"

rawRdd_train = sc.textFile(pathFile).map(lambda line : line.split('\t'))
print('finish read rdd, start to init action log rdd:')
actionLogRdd_train = rawRdd_train.map(
    lambda x :(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
               int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))
sqlContext = SQLContext(sc)
labels=[('uid',typ.IntegerType()),
    ('user_city',typ.IntegerType()),
    ('item_id',typ.IntegerType()),
    ('author_id',typ.IntegerType()),
    ('item_city',typ.IntegerType()),
    ('channel',typ.IntegerType()),
    ('finish',typ.IntegerType()),
    ('like',typ.IntegerType()),
    ('music_id',typ.IntegerType()),
    ('device',typ.IntegerType()),
    ('time',typ.LongType()),
    ('duration_time',typ.IntegerType())]
actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])

df_origin = sqlContext.createDataFrame(actionLogRdd_train, actionLogSchema)

df_origin_1=df_origin


def add_index(df):
    idx = 0
    print(df.count())
    index_list = [x for x in range(1, df.count()+1)]
      # 构造一个列表存储索引值，用生成器会出错
    # 定义一个函数
    def set_index(x):
        global idx    # 将idx设置为全局变量
        if x is not None:
            idx += 1
            return index_list[idx-1]
    index = udf(set_index, typ.IntegerType())    # udf的注册，这里需要定义其返回值类型
    df1=df.select(fn.col("*"), index("item_id").alias("id"))  # udf的注册的使用，alias方法用于修改列名,报错TypeError: 'str' object is not callable
    #df1.show()
    return df1

feature_index_train=add_index(df_origin)
feature_index_train.show(5,truncate=False)

feature_index_train1=add_index(df_origin_1)
feature_index_train1.show(5,truncate=False)


df_400= df_origin.limit(40).repartition(10)
print(df_400.rdd.getNumPartitions())
df=df_400.limit(20).repartition(10)
print(df.rdd.getNumPartitions())
print(df_400.count())
df1=df_400.subtract(df)
print(df1.count())
print(df1.rdd.getNumPartitions())


df.show(30)
df2=df.coalesce(1).withColumn("id", monotonically_increasing_id())

df1.show(30)
df3=df1.coalesce(1).withColumn("id", monotonically_increasing_id())
print("观察repartition(1)之后，数据df2.limit(30)和df3.limit(30)是否相同")
df2.limit(30).show(30)

df3.limit(30).show(30)

'''
time_min = df.select(fn.min(df['time'])).collect()
print(time_min)
print(time_min[0][0])


df=df.withColumn('time_day', ((df.time-fn.lit(time_min[0][0])) /fn.lit(3600 * 24)).cast(typ.IntegerType()))


print("计算各特征的count，包括交叉特征")
count_feats_list = []
count_feats_list.append(['time_day'])

print('single feature count')
count_feats_list.extend([[c] for c in df.columns if c not in ['time', 'channel', 'like', 'finish']])
print(count_feats_list)

print('cross count')
users = ['uid']
# authors = ['user_city', 'author_id', 'item_city', 'channel', 'music_id', 'device', 'duration_time']
authors = ['item_id', 'user_city', 'author_id', 'item_city', 'channel', 'music_id', 'device', 'duration_time']
count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

users = ['author_id']
# authors = ['item_city', 'music_id', 'duration_time', 'time_day']
authors = ['channel', 'user_city', 'item_city', 'music_id', 'duration_time', 'time_day']
count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

count_feats_list.append(['uid', 'user_city', 'channel', 'device'])
count_feats_list.append(['author_id', 'item_city', 'music_id', 'duration_time'])
print("计算count的字段有以下这些")
print(count_feats_list)



res_list=[]
def count_fun(data, group_cols):
    new_feature = '_'.join(group_cols) + '_count'
    data[new_feature] = ""
    for c in group_cols:
        data[new_feature] = data[new_feature] + '_' + data[c].astype(str)   #值的拼接
    count = Counter(data[new_feature])   #统计各特征下每个值出现的次数
    data[new_feature] = data[new_feature].apply(lambda x: count[x])
    return data[[new_feature]]           #各特征下的值转化为每个值出现的次数

# df=df.withColumn('uid_item_id', fn.concat_ws('_',df["uid"].cast(typ.StringType()),df["item_id"].cast(typ.StringType())))
# df.show(5,truncate=False)
#['uid', 'item_id']如计算这两个字段的count
for i in range(len(count_feats_list)):
   group_cols=count_feats_list[i]
   new_feature = '_'.join(group_cols)
   #判断是几维交叉特征，并进行拼接，再计算每个特征值的个数count,并完成映射
   if len(group_cols)==1:
      df1 = df.groupby(new_feature).count()\
              .withColumnRenamed('count',new_feature+'_count')
      df=df.join(df1,new_feature,'left')

      #类别偏好的ratio比例
      # count_feats = (count_feats - count_feats.min()) / (count_feats.max() - count_feats.min())
      count_min = df.select(fn.min(df[new_feature+'_count'])).collect()[0][0]
      count_max = df.select(fn.max(df[new_feature+'_count'])).collect()[0][0]
      df=df.withColumn(new_feature+'_count_ratio', ((df[new_feature+'_count']-fn.lit(count_min)) /(fn.lit(count_max)-fn.lit(count_min))).cast(typ.IntegerType()))



   if len(group_cols)==2:
      df=df.withColumn(new_feature, fn.concat_ws('_',df[group_cols[0]].cast(typ.StringType()),df[group_cols[1]].cast(typ.StringType()))
                                                     )
      df2 = df.groupby(new_feature).count()\
             .withColumnRenamed('count',new_feature+'_count')
      df=df.join(df2,new_feature,'left')

   if len(group_cols)==4:
      df=df.withColumn(new_feature, fn.concat_ws('_',df[group_cols[0]].cast(typ.StringType()),df[group_cols[1]].cast(typ.StringType()),
                                                     df[group_cols[2]].cast(typ.StringType()),df[group_cols[3]].cast(typ.StringType()))
                                                   )
      df3 = df.groupby(new_feature).count()\
             .withColumnRenamed('count',new_feature+'_count')
      df=df.join(df3,new_feature,'left')

df.show(5,truncate=False)
# for i in range(len(count_feats_list)):
#     res_list.append(p.apply_async(count_fun,
#                                   args=(data[count_feats_list[i]], count_feats_list[i])))
#     print(str(i) + ' processor started !')
'''