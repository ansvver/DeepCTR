___author__ = 'zlx'
#!/usr/local/bin/python
# path = '/data/code/DeepCTR/featureEngineering/spark'#all_functions.py的所在路径
# sys.path.append(path)
import  configparser
import os
import sys
from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
import datetime
from pyspark.sql.functions import udf
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import gc
from collections import Counter
# workspace_path='/data/code/DeepCTR/'
# hdfs_data_path = '/user/hadoop/icmechallenge2019/track2/data/'

class SparkFEProcess:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark of explore_spark_step3") \
            .set("spark.ui.showConsoleProgress", "false")
        self.sc = SparkContext(conf=sparkConf)
        self.sc.broadcast(self.parser)
        self.init_logger()
        # #初始化相关参数
        # #bins_dict保存相关列的分箱方案，在处理测试数据的时候使用
        # self.bins_dict={}


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

    def data_describe(self):
        sqlContext = SQLContext(self.sc)
        print('starto read data after explore_saprk_step1_cross:')
        rootPath=self.parser.get("hdfs_path", "hdfs_data_path")
        print('start to read actLog_train_single_cross')
        test_file_path = rootPath + 'actLog_test_single_cross'
        actLog_test_rdd = self.sc.pickleFile(test_file_path)
        #比对label，看labels是否合适
        labels=[  ('duration_time',typ.IntegerType()),
                ('device',typ.IntegerType()),
                ('music_id',typ.IntegerType()),
                ('item_city',typ.IntegerType()),
                ('author_id',typ.IntegerType()),
                ('item_id',typ.IntegerType()),
                ('user_city',typ.IntegerType()),
                ('uid',typ.IntegerType()),
                ('channel',typ.IntegerType()),
                ('finish',typ.IntegerType()),
                ('like',typ.IntegerType()),
                ('time_day',typ.IntegerType()),
                ('item_pub_month',typ.IntegerType()),
                ('item_pub_day',typ.LongType()),
                ('item_pub_hour',typ.IntegerType()),
                ('item_pub_minute',typ.IntegerType()),
                ('uid_count_bin',typ.IntegerType()),
                ('user_city_count_bin',typ.IntegerType()),
                ('user_city_count_ratio',typ.DoubleType()),
                ('item_id_count_bin',typ.IntegerType()),
                ('item_id_count_ratio',typ.DoubleType()),
                ('author_id_count_bin',typ.IntegerType()),
                ('author_id_count_ratio',typ.DoubleType()),
                ('item_city_count_bin',typ.IntegerType()),
                ('item_city_count_ratio',typ.DoubleType()),
                ('music_id_count_bin',typ.IntegerType()),
                ('music_id_count_ratio',typ.DoubleType()),
                ('device_count_bin',typ.IntegerType()),
                ('device_count_ratio',typ.DoubleType()),
                ('uid_author_id_count_bin',typ.IntegerType()),
                ('uid_author_id_count_ratio',typ.DoubleType()),
                 ('uid_item_city_count_bin',typ.IntegerType()),
                ('uid_item_city_count_ratio',typ.DoubleType()),
                ('uid_channel_count_bin',typ.IntegerType()),
                ('uid_channel_count_ratio',typ.DoubleType()),
                ('uid_music_id_count_bin',typ.IntegerType()),
                ('uid_music_id_count_ratio',typ.DoubleType()),
                ('uid_device_count_bin',typ.IntegerType()),
                ('uid_device_count_ratio',typ.DoubleType()),
                ('author_id_channel_count_bin',typ.IntegerType()),
                ('author_id_channel_count_ratio',typ.DoubleType()),
                ('author_id_user_city_count_bin',typ.IntegerType()),
                ('author_id_user_city_count_ratio',typ.DoubleType()),
                ('author_id_item_city_count_bin',typ.IntegerType()),
                ('author_id_item_city_count_ratio',typ.DoubleType()),
                ('author_id_music_id_count_bin',typ.IntegerType()),
                ('author_id_music_id_count_ratio',typ.DoubleType()),
                ('uid_channel_device_count_bin',typ.IntegerType()),  #改成uid_channel_device
                ('uid_channel_device_count_ratio',typ.DoubleType()),  #改成uid_channel_device
                ('author_id_item_city_music_id_count_bin',typ.IntegerType()),
                ('author_id_item_city_music_id_count_ratio',typ.DoubleType()),
            ]
        actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])

        df_actLog_test = sqlContext.createDataFrame(actLog_test_rdd,actionLogSchema)
        # df_actLog_test.show(1,truncate=False)

        print('start to read actLog_train_single_cross')
        train_file_path = rootPath + 'actLog_train_single_cross'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        df_actLog_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)
        # df_actLog_train.show(1,truncate=False)


        return df_actLog_train, df_actLog_test




    def data_explore(self,df_train,df_test):

        sqlContext = SQLContext(self.sc)

        print("对item_pub_hour进行离散化")
        def hourBin(x):
            if x>=23 or x <=2:
                return 1
            elif 3<=x<8:
                return 2
            elif 8<=x<12:
                return 3
            else:
                return 4

        converHourBin=udf(lambda x :hourBin(x), typ.IntegerType())
        df_train = df_train.withColumn("item_pub_hour", converHourBin(df_train.item_pub_hour))
        df_test = df_test.withColumn("item_pub_hour", converHourBin(df_test.item_pub_hour))

        print("----1、计算统计特征：用户特征和item特征之间的条件概率---------")
        feats_list = []

        condition = ['uid']
        authors = ['music_id','item_pub_hour']  #'author_id', 'item_city', 'channel',
        feats_list.extend([[u_col, a_col] for u_col in condition for a_col in authors])
        df_tmp=df_train.select(condition)
        df2=df_tmp.groupby(condition).count().withColumnRenamed('count',condition[0]+'_count')
        # df2.show(1,truncate=False) # ['uid','uid_count']
        df2.cache()
        # df_train=df_train.join(df2,condition,'left')
        # df_train.show(1,truncate=False)
        # cannot resolve '`uid_count`' given input columns: [time, user_city, like, author_id, uid, device, music_id, finish, duration_time, channel, item_city, item_id]
        # del df2
        # gc.collect()
        for feature_group in feats_list:
            print(feature_group+[feature_group[0]+'_count'])   #+[feature_group[0]+'_count']
            df1=df_train.select(feature_group).groupby(feature_group).count()
            # df1.show(1,truncate=False)   #理论上还是只有3个字段，不包含uid_count
            df1=df1.join(df2,condition,'left')
            df1.show(1,truncate=False)   #|uid|item_pub_hour|count|uid_count
            df1=df1.withColumn(feature_group[1]+'_'+feature_group[0]+"_condition_ratio",fn.col('count')/fn.col(feature_group[0]+'_count'))
            df1=df1.drop('count').drop(feature_group[0]+'_count')
            df1.show(1,truncate=False)
            print(df_train.columns)
            print(df1.columns)
            df_train=df_train.join(df1,feature_group,"left")   #|uid|item_pub_hour|item_pub_hour_uid_condition_ratio
            df_train.show(1,truncate=False)
            df_test=df_test.join(df1,feature_group,"left").na.fill({feature_group[1]+'_'+feature_group[0]+"_condition_ratio":0})  #对某一列填充缺失值
            df_test.show(1,truncate=False)



        feats_list = []
        condition = ['item_id']
        authors = ['uid_city', 'channel']
        feats_list.extend([[u_col, a_col] for u_col in condition for a_col in authors])

        df_tmp=df_train.select(condition)
        df2=df_tmp.groupby(condition).count().withColumnRenamed('count',condition[0]+'_count')
        # df2.show(1,truncate=False) # ['uid','uid_count']
        df2.cache()
        # df_train=df_train.join(df2,condition,'left')
        # df_train.show(1,truncate=False)
        # cannot resolve '`uid_count`' given input columns: [time, user_city, like, author_id, uid, device, music_id, finish, duration_time, channel, item_city, item_id]
        # del df2
        # gc.collect()
        for feature_group in feats_list:
            print(feature_group+[feature_group[0]+'_count'])   #+[feature_group[0]+'_count']
            df1=df_train.select(feature_group).groupby(feature_group).count()
            # df1.show(1,truncate=False)   #理论上还是只有3个字段，不包含uid_count
            df1=df1.join(df2,condition,'left')
            df1.show(1,truncate=False)
            df1=df1.withColumn(feature_group[1]+'_'+feature_group[0]+"_condition_ratio",fn.col('count')/fn.col(feature_group[0]+'_count'))
            df1=df1.drop('count').drop(feature_group[0]+'_count')
            # df1.show(5)
            df_train=df_train.join(df1,feature_group,"left")
            df_train.show(1,truncate=False)
            df_test=df_test.join(df1,feature_group,"left").na.fill({feature_group[1]+'_'+feature_group[0]+"_condition_ratio":0})  #对某一列填充缺失值
            df_test.show(1,truncate=False)


        df_train=df_train.drop('uid_count').drop('item_id_count')
        df_train.printSchema()
        df_test.printSchema()

        print('-------5.保存数据预处理结果-------')
        test_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_test_step3_try'
        os.system("hadoop fs -rm -r {}".format(test_file_path))
        df_test.rdd.map(tuple).saveAsPickleFile(test_file_path)

        del df_test
        gc.collect()

        train_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_train_step3_try'
        os.system("hadoop fs -rm -r {}".format(train_file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train.rdd.map(tuple).saveAsPickleFile(train_file_path)


        #观察这波特征如何与


if __name__ == "__main__":
    spark_job = SparkFEProcess()

    df_train,df_test=spark_job.data_describe()

    spark_job.data_explore(df_train,df_test)
