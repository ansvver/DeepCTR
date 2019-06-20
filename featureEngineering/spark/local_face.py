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
import json

conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

pathFile="C:/data/douyinData/track2_face_attrs_500.txt"

print('start to read data for rdd:')
sqlContext = SQLContext(sc)
df = sqlContext.read.json(pathFile)
df=df.select("item_id",fn.explode("face_attrs")).toDF("item_id","face_attrs")
df = df.select(df["item_id"].cast(typ.IntegerType()),"face_attrs.beauty", df["face_attrs.gender"].cast(typ.IntegerType()),"face_attrs.relative_position")
attrs = sc.parallelize(["relative_position_" + str(i) for i in range(4)]).zipWithIndex().collect()
print("列名：", attrs)
for name, index in attrs:
    df = df.withColumn(name, df['relative_position'].getItem(index))
#删除 relative_position
df =df.drop('relative_position')

print("df")
df.show(5,truncate=False)
df.printSchema()



#!/usr/local/bin/python
# path = '/data/code/DeepCTR/featureEngineering/spark'#all_functions.py的所在路径
# sys.path.append(path)
import  configparser
import os
import json
import sys
from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.ml.feature import VectorAssembler
import datetime
from pyspark.sql.functions import udf
import pyspark.sql.functions  as psf
from pyspark.ml.linalg import Vectors, VectorUDT,SparseVector
import numpy as np
import gc
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Normalizer
import jsonpath

class SparkFEProcess:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark") \
            .set("spark.ui.showConsoleProgress", "false")
        self.sc = SparkContext(conf=sparkConf)
        self.sc.broadcast(self.parser)
        self.init_logger()


    # def init_config(self):
    #     config_file = workspace_path + 'resource/config.ini'
    #     parser = configparser.ConfigParser()
    #     parser.read(config_file)
    #     return  parser
    def init_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        workspace_path = current_path.split('featureEngineering')[0]
        config_file = workspace_path + 'resource/config.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file)
        return  parser

    def init_logger(self):
        '''
        设置日志级别
        :param sc:
        :return:
        '''
        logger = self.sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
        logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
        logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)


    def read_rdd(self, fileName):
        try:
            file_path = self.parser.get("hdfs_path", "hdfs_data_path") + fileName
            data_rdd = self.sc.textFile(file_path)
            return data_rdd
        except Exception as e:
            print(e)

    def read_json(self,dic):
        key=['item_id','gender','beauty','relative_position']
        line={}
        for k in key:
            s = jsonpath.jsonpath(dic,'$..'+k)  #如果没有会返回false
            if s==False:
               line[k]=-1
            else:
               line[k]=s[0]
        return line

    def data_describe(self):
        print('start to read data for rdd:')
        rawRdd_face = self.read_rdd('track2_face_attrs.txt').map(lambda line : json.loads(line))
        # rawRdd_face.cache()
        global keys
        keys=['item_id','gender','beauty','relative_position']
        rawRdd_face2=rawRdd_face.map(lambda dic:{key :jsonpath.jsonpath(dic,'$..'+key)[0] if jsonpath.jsonpath(dic,'$..'+key) else None  for key in keys})
        print(rawRdd_face2.take(10))
        #转化为dataframe,在不指定schema的情况下会自动推断
        sqlContext = SQLContext(self.sc)
        labels=[
            ('item_id',typ.IntegerType()),
            ('gender',typ.IntegerType()),
            ('beauty',typ.FloatType()),
            ('relative_position',typ.ArrayType(typ.FloatType()))]
        Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df = sqlContext.createDataFrame(rawRdd_face2,Schema)
        # df.show()
        # df.printSchema()

        attrs = self.sc.parallelize(["relative_position_" + str(i) for i in range(4)]).zipWithIndex().collect()
        print("列名：", attrs)
        for name, index in attrs:
            df = df.withColumn(name, df['relative_position'].getItem(index))
        #删除 relative_position
        df =df.drop('relative_position')
        '''
        print('查看各列的数据分布情况,最大值、最小值、是否存在0值')
        df.describe().show()

        print('beauty列中为0的行数')
        print(df.filter(df['beauty']==0).count())
        print('relative_position_0列中为0的行数')
        print(df.filter(df['relative_position_0']==0).count())
        print('relative_position_1列中为0的行数')
        print(df.filter(df['relative_position_1']==0).count())
        print('relative_position_2列中为0的行数')
        print(df.filter(df['relative_position_2']==0).count())
        print('relative_position_3列中为0的行数')
        print(df.filter(df['relative_position_3']==0).count())

        #空值是一定要填充的，但是填充-1是否可以？
        print('检查beauty是否存在空值')
        df1 = df.filter(psf.isnull("beauty"))  # 把a列里面数据为null的筛选出来（代表python的None类型）
        df1.show(5)

        print('检查relative_position_0是否存在空值,与relative_position_1，2,3等同，查看其中一列即可，要么同时存在，要么同时不存在')
        df3 = df.filter(psf.isnull("relative_position_0"))  # 把a列里面数据为null的筛选出来（代表python的None类型）
        df3.show(5)


        print('查看每一列的缺失比例')  #这里的话face_attrs各属性的缺失比例是一样的
        df.agg(*[(1-(psf.count(c) /psf.count('*'))).alias(c+'_missing') for c in df.columns]).show()

        #查看每行的缺失值
        print('查看每行记录的缺失值')
        print(df.rdd.map(lambda row:(row['item_id'],sum([c==None for c in row]))).toDF('item_id','sum_none').filter('sum_none'>0).count())
        '''
        # 类别变量缺失值填充为-1,连续变量缺失值填充为均值
        print('输出各均值')
        mean_val = df.select(psf.mean(df['beauty'])).collect()
        mean_beauty = mean_val[0][0] # to show the number
        print(mean_beauty)
        mean_val = df.select(psf.mean(df['relative_position_0'])).collect()
        mean_relative_position0 = mean_val[0][0] # to show the number
        print(mean_relative_position0)
        mean_val = df.select(psf.mean(df['relative_position_1'])).collect()
        mean_relative_position1 = mean_val[0][0] # to show the number
        print(mean_relative_position1)
        mean_val = df.select(psf.mean(df['relative_position_2'])).collect()
        mean_relative_position2 = mean_val[0][0] # to show the number
        print(mean_relative_position2)
        mean_val = df.select(psf.mean(df['relative_position_3'])).collect()
        mean_relative_position3 = mean_val[0][0] # to show the number
        print(mean_relative_position3)

        df=df.na.fill({'gender': -1, 'beauty': mean_beauty,'relative_position_0': mean_relative_position0, \
                       'relative_position_1': mean_relative_position1,'relative_position_2': mean_relative_position2,\
                       'relative_position_3': mean_relative_position3})

        print('填充缺失以后')
        df.show(2,truncate=False)
        # print(df.columns)


        print('-------5.保存数据预处理结果-------')
        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'face_feature'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df.rdd.map(tuple).saveAsPickleFile(file_path)

        print('数据保存结束')







if __name__ == "__main__":
    spark_job = SparkFEProcess()

    spark_job.data_describe()








