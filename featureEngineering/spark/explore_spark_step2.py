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

        sparkConf = SparkConf().setAppName("feature engineering on spark of explore_spark_2") \
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
        df_actLog_test.show(1,truncate=False)

        print('start to read actLog_train_single_cross')
        train_file_path = rootPath + 'actLog_train_single_cross'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        df_actLog_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)
        df_actLog_train.show(1,truncate=False)


        return df_actLog_train, df_actLog_test

    def data_explore(self,df_train,df_test):
        sqlContext = SQLContext(self.sc)
        print("--------1、针对uid，authorid，musicid等组合的正负样本数量统计特征--------")
        print("交叉特征的正负样本数量统计")
        posneg_feats_list = []
        # posneg_feats_list.append(["duration_time"])
        # posneg_feats_list.append(["time_day"])
        print('cross count')
        users = ['uid']
        authors = ['author_id', 'item_city', 'channel', 'music_id','item_pub_hour']

        posneg_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])
        posneg_feats_list.append(['uid','author_id', 'channel'])
        posneg_feats_list.append(['uid', 'author_id', 'music_id'])
        # posneg_feats_list.append(['uid','author_id', 'channel','time_day'])
        # posneg_feats_list.append(['uid', 'author_id', 'music_id','time_day'])

        print("计算以下交叉特征的正负样本比例")  #有2、3、4维的交叉特征
        print(posneg_feats_list)

        for i in range(len(posneg_feats_list)):
            group_cols=posneg_feats_list[i]
            new_feature = '_'.join(group_cols)
            #计算df_train数据中正负样本的比例，test中直接拼接，为null则填充为0或者均值
            #正负样本判定字段：like  finish
            #d第一步，先拼接
            print(new_feature)
            if len(group_cols)==2:
                print("开始处理2维交叉变量")
                df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
                                                                 )
                df_test=df_test.withColumn(new_feature, fn.concat_ws('_',df_test[group_cols[0]].cast(typ.StringType()),df_test[group_cols[1]].cast(typ.StringType()))
                                                                 )

            if len(group_cols)==3:

                print("开始处理3维交叉变量")
                df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
                                                                 ,df_train[group_cols[2]].cast(typ.StringType()))
                df_test=df_test.withColumn(new_feature, fn.concat_ws('_',df_test[group_cols[0]].cast(typ.StringType()),df_test[group_cols[1]].cast(typ.StringType()))
                                                                 ,df_test[group_cols[2]].cast(typ.StringType()))
            # if len(group_cols)==4:
            #
            #     print("开始处理4维交叉变量")
            #     df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
            #                                                      ,df_train[group_cols[2]].cast(typ.StringType()) ,df_train[group_cols[3]].cast(typ.StringType()))
            #     df_test=df_test.withColumn(new_feature, fn.concat_ws('_',df_test[group_cols[0]].cast(typ.StringType()),df_test[group_cols[1]].cast(typ.StringType()))
            #                                                      ,df_test[group_cols[2]].cast(typ.StringType()) ,df_test[group_cols[3]].cast(typ.StringType()))

            for target in ["like","finish"] :
                df3=df_train.groupby(new_feature).count().withColumnRenamed('count',new_feature+'_count')
                df4=df_train.where(df_train[target]==1).groupby(new_feature).count().withColumnRenamed('count',new_feature+"_count_"+target+"_1")
                df3=df3.join(df4,new_feature,'left').na.fill(0)
                del df4
                gc.collect()
                # print("两列相除:得到正样本的比例",target)
                df3=df3.withColumn(new_feature+"_"+target+"_pos_neg",fn.col(new_feature+"_count_"+target+"_1")/fn.col(new_feature+'_count'))
                df3=df3.drop(new_feature+"_count_"+target+"_1",new_feature+'_count')
                print("新的df_train",new_feature,target)
                df_train=df_train.join(df3,new_feature,"left")
                df_train.show(1)
                df_test=df_test.join(df3,new_feature,"left") #会存在null，缺失值设置为0
                print("新的df_test",new_feature,target)
                df_test.show(1)
                df_test=df_test.na.fill(0)
                del df3
                gc.collect()
            if new_feature not in ["duration_time","time_day"]:
                df_train=df_train.drop(new_feature)
                df_test=df_test.drop(new_feature)
                df_train.printSchema()
                df_test.printSchema()

        print('最终表结构，该表结构用于concate的输入')   #是不是应该有build_data_for_like  build_data_for_finish
        df_train.printSchema()
        df_test.printSchema()

        print("查看test缺失值")
        df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()
        print("查看train缺失值")
        df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()

        print('-------5.保存数据预处理结果-------')
        test_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_test_step2'
        os.system("hadoop fs -rm -r {}".format(test_file_path))
        df_test.rdd.map(tuple).saveAsPickleFile(test_file_path)

        del df_test
        gc.collect()

        train_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_train_step2'
        os.system("hadoop fs -rm -r {}".format(train_file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train.rdd.map(tuple).saveAsPickleFile(train_file_path)





if __name__ == "__main__":
    spark_job = SparkFEProcess()

    df_train,df_test=spark_job.data_describe()

    spark_job.data_explore(df_train,df_test)
