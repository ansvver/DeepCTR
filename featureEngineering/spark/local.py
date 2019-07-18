# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

pathFile="D:/douyinData/train_1w.txt"
# pathFile="D:/douyinData/final_track2_train.txt"
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

df_train = sqlContext.createDataFrame(actionLogRdd_train, actionLogSchema)

feature_group=['uid','author_id']
df_tmp=df_train.select(feature_group)
df1=df_tmp.groupby(feature_group).count()
df1.show(5)
df2=df_tmp.groupby(feature_group[0]).count().withColumnRenamed('count',feature_group[0]+'_count')
df2.show(5)

df1=df1.join(df2,feature_group[0],'left')
df1.show(5)
df1=df1.withColumn(feature_group[1]+'_'+feature_group[0]+"_condition_ratio",fn.col('count')/fn.col(feature_group[0]+'_count'))
df1=df1.drop()
df1.show(5)

#

