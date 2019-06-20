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

        sparkConf = SparkConf().setAppName("feature engineering on spark of face") \
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

        attrs = self.sc.parallelize(["relative_position_" + str(i) for i in range(4)]).zipWithIndex().collect()
        print("列名：", attrs)
        for name, index in attrs:
            df = df.withColumn(name, df['relative_position'].getItem(index))
        #删除 relative_position
        df_face =df.drop('relative_position')
        del df
        gc.collect()


        # print('-------保存df_face数据-------')
        # file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'face_feature'
        # os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        # df_face.rdd.map(tuple).saveAsPickleFile(file_path)
        # print('数据保存结束')

        print('start to read act data  only for uid and item_id :')
        rawRdd_train = self.read_rdd('final_track2_train.txt').map(lambda line : line.split('\t'))
        rawRdd_test = self.read_rdd('final_track2_test_no_anwser.txt').map(lambda line : line.split('\t'))
        actionLogRdd_train = rawRdd_train.map(
            lambda x :(int(x[0]), int(x[2])))
        # total = actionLogRdd_train.count()
        # print('total: ' + str(total))
        actionLogRdd_test = rawRdd_test.map(
            lambda x :(int(x[0]), int(x[2])))

        sqlContext = SQLContext(self.sc)
        labels=[('uid',typ.IntegerType()),
            ('item_id',typ.IntegerType())
            ]

        actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])

        dfactionLog_train = sqlContext.createDataFrame(actionLogRdd_train, actionLogSchema)
        dfactionLog_test = sqlContext.createDataFrame(actionLogRdd_test, actionLogSchema)

        #根据item_id进行关联
        df_face=df_face.select(["item_id","gender","beauty"])
        df_uid_face_test=dfactionLog_test.select(["uid","item_id"]).join(df_face,'item_id','left').drop("item_id")
        df_uid_face_train=dfactionLog_train.select(["uid","item_id"]).join(df_face,'item_id','left').drop("item_id")
        del dfactionLog_test
        del dfactionLog_train
        gc.collect()

        #进行处理
        gdf=df_uid_face_train.groupby("uid")
        df1=gdf.agg(fn.max("beauty").alias("uid_max_beauty"),fn.avg("beauty").alias("uid_avg_beauty"),(fn.sum("gender")/fn.count("gender")).alias("uid_male_ratio"))
        df1.show(1,truncate=False)
        df_uid_face_train=df_uid_face_train.join(df1,'uid','left').drop("gender").drop("beauty")
        df_uid_face_test=df_uid_face_test.join(df1,'uid','left').drop("gender").drop("beauty")

        print("理论上应该只有uid，uid_max_beauty,uid_avg_beauty,uid_male_ratio")
        df_uid_face_train.printSchema()
        df_uid_face_test.printSchema()



        print('-------保存df_uid_face数据-------')
        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'df_uid_face_train'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_uid_face_train.rdd.map(tuple).saveAsPickleFile(file_path)

        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'df_uid_face_test'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_uid_face_test.rdd.map(tuple).saveAsPickleFile(file_path)
        print('数据保存结束')












if __name__ == "__main__":
    spark_job = SparkFEProcess()

    spark_job.data_describe()



    # spark_job.data_explore(dfactionLog_train,dfactionLog_test)