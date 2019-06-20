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

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

pathFile="C:/data/douyinData/track2_title_500.txt"

print('start to read data for rdd:')
rawRdd_nlp = sc.textFile(pathFile).map(lambda line : eval(line))
# print(rawRdd_nlp.take(10))
#转化为dataframe,在不指定schema的情况下会自动推断
sqlContext = SQLContext(sc)
labels=[
    ('item_id',typ.IntegerType()),
    ('title_features',typ.MapType(typ.StringType(), typ.IntegerType()))]
Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
df = sqlContext.createDataFrame(rawRdd_nlp,Schema)

print("统计title中不同词的个数unique，以及title的长度")
gdf=df.select("item_id",fn.explode(fn.col("title_features"))).groupBy("item_id")
df2=gdf.agg(fn.count("key").alias("title_words_unique"))

df3=gdf.agg(fn.sum("value").alias("title_length"))

df=df.join(df2,"item_id","left") \
     .join(df3,"item_id","left")
df.printSchema()




