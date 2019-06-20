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

        sparkConf = SparkConf().setAppName("feature engineering on spark of explore_spark_cross") \
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
        rootPath=self.parser.get("hdfs_path", "hdfs_data_path")
        print('start to read actLog_single  ,and to deal with cross_feature')
        train_file_path = rootPath + 'actLog_train_single'
        test_file_path  = rootPath + 'actLog_test_single'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        actLog_test_rdd = self.sc.pickleFile(test_file_path)
        #修改label

        labels=[('duration_time',typ.IntegerType()),
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
                ('music_id_count_bin:',typ.IntegerType()),
                ('music_id_count_ratio',typ.DoubleType()),
                ('device_count_bin',typ.IntegerType()),
                ('device_count_ratio',typ.DoubleType()),
                ('duration_time_count_bin',typ.IntegerType()),  #由于step1_single中多处理这个字段，这个字段其实用不上，读进来后删掉
                ('duration_time_count_ratio',typ.DoubleType())
                ]

        actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df_actLog_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)
        df_actLog_test = sqlContext.createDataFrame(actLog_test_rdd,actionLogSchema)
        df_actLog_train=df_actLog_train.drop('duration_time_count_bin').drop('duration_time_count_ratio')
        df_actLog_test=df_actLog_test.drop('duration_time_count_bin').drop('duration_time_count_ratio')

        # df_actLog_train.show(5,truncate=False)
        df_actLog_train.printSchema()
        # df_actLog_test.show(5,truncate=False)
        df_actLog_test.printSchema()

        return df_actLog_train, df_actLog_test


    def bining(self,sqlContext,df,col,percent_list):
        '''
        :param sqlContext:
        :param df:
        :param col:  需要分箱的列
        :return:
        '''
        pandas_df = df.toPandas()
        bins=[]
        for percent in percent_list:
            bins.append(np.percentile(pandas_df.loc[:,col],percent))  #至少有20%的数据项小于或等于这个值
        print(col+'查看分箱')
        print(bins)
        pandas_df.loc[:,col]=np.digitize(pandas_df.loc[:,col],bins,right=True)
        #修改pandas中的列名
        pandas_df.rename(columns={col:col+'_bin'}, inplace = True)
        df_spark= sqlContext.createDataFrame(pandas_df)
        # df_spark.show()
        return  df_spark

    # def city_col_deal(self,df,col):
    #     df_city_score=df.groupBy(col).avg('finish', 'like') \
    #         .withColumnRenamed("avg(finish)","avg_finish").withColumnRenamed("avg(like)","avg_like")
    #     df_city_score=df_city_score.withColumn(col+'_score', df_city_score.avg_finish*0.7+df_city_score.avg_like*0.3)\
    #                           .select(col,fn.bround(col+'_score', scale=4).alias(col+'_score'))
    #     return df_city_score

    def dropUnuseCols(self,df,unuse_col):
        for col in unuse_col:
            df=df.drop(col)
        return df


    def data_explore(self,df_train,df_test):

        sqlContext = SQLContext(self.sc)

        print('--------2、统计特征：count、ratio、nunique、ctr相关特征')
        print("计算基础特征和交叉特征的count、类别偏好的ratio")
        count_feats_list = []
        print('cross count')
        users = ['uid']
        authors = ['author_id', 'item_city', 'channel', 'music_id', 'device']
        count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

        users = ['author_id']
        authors = ['channel', 'user_city', 'item_city', 'music_id']
        count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

        count_feats_list.append(['uid', 'channel', 'device'])
        count_feats_list.append(['author_id', 'item_city', 'music_id'])
        print("计算count的字段有以下这些")
        print(count_feats_list)

        for i in range(len(count_feats_list)):
           group_cols=count_feats_list[i]
           new_feature = '_'.join(group_cols)
           print("根据上述保存的df_train 和df_test 再处理2维交叉变量")
           if len(group_cols)==2:
              print("开始处理2维交叉变量")
              df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()))
                                                             )
              df_test=df_test.withColumn(new_feature, fn.concat_ws('_',df_test[group_cols[0]].cast(typ.StringType()),df_test[group_cols[1]].cast(typ.StringType()))
                                                             )
              df2 = df_train.groupby(new_feature).count()\
                     .withColumnRenamed('count',new_feature+'_count')
              #类别偏好的ratio比例
              count_min = df2.select(fn.min(df2[new_feature+'_count'])).collect()[0][0]
              count_max = df2.select(fn.max(df2[new_feature+'_count'])).collect()[0][0]
              # F.bround("Rank", scale=4)
              df2=df2.withColumn(new_feature+'_count_ratio', fn.bround(((df2[new_feature+'_count']-fn.lit(count_min)) /((fn.lit(count_max)-fn.lit(count_min)).cast(typ.IntegerType()))),scale=3))

              if  new_feature=="uid_author_id":     #用户看了这个用户发布的视频 超过2个
                  percent_list=[0,90,95,98,100]
              if  new_feature=="uid_music_id":
                  percent_list=[0,75,90,95,98,100]
              if  new_feature=="uid_device":
                  percent_list=[0,25,50,75,90,100]
              if  new_feature=="author_id_user_city":
                  percent_list=[0,75,90,95,98,100]
              if  new_feature=="author_id_music_id":
                  percent_list=[0,75,90,95,98,100]
              else:
                 percent_list=[0,50,75,90,95,100]

              df2=self.bining(sqlContext,df2,new_feature+'_count',percent_list)
              print("查看df2_2")
              df2.show(1,truncate=False)
              df_train=df_train.join(df2,new_feature,'left').drop(new_feature)
              print("train")
              df_train.show(1,truncate=False)   #ratio是一个连续变量，范围0-1
              df_train.printSchema()
              df_test=df_test.join(df2,new_feature,'left').drop(new_feature)    #先关联后删除
              print("test")
              df_test.show(1,truncate=False)


           if len(group_cols)==3:
              print("开始处理3维交叉变量")
              df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()),
                                                             df_train[group_cols[2]].cast(typ.StringType()))
                                                           )
              df_test=df_test.withColumn(new_feature, fn.concat_ws('_',df_test[group_cols[0]].cast(typ.StringType()),df_test[group_cols[1]].cast(typ.StringType()),
                                                             df_test[group_cols[2]].cast(typ.StringType()))
                                                           )

              df3 = df_train.groupby(new_feature).count()\
                     .withColumnRenamed('count',new_feature+'_count')

              #类别偏好的ratio比例
              count_min = df3.select(fn.min(df3[new_feature+'_count'])).collect()[0][0]
              count_max = df3.select(fn.max(df3[new_feature+'_count'])).collect()[0][0]
              # F.bround("Rank", scale=4)
              df3=df3.withColumn(new_feature+'_count_ratio', fn.bround(((df3[new_feature+'_count']-fn.lit(count_min)) /((fn.lit(count_max)-fn.lit(count_min)).cast(typ.IntegerType()))),scale=3))
              # print("查看df3_1")
              # df3.show(5,truncate=False)
              percent_list=[0,50,75,90,95,100]
              df3=self.bining(sqlContext,df3,new_feature+'_count',percent_list)
              print("查看df3_2")
              df3.show(1,truncate=False)
              df_train=df_train.join(df3,new_feature,'left').drop(new_feature)
              print("train")
              df_train.show(1,truncate=False)
              df_train.printSchema()
              df_test=df_test.join(df3,new_feature,'left').drop(new_feature)
              print("test")
              df_test.show(1,truncate=False)


        print("交叉特征处理结束")
        print("查看train的表结构")
        df_train.printSchema()
        # print("删除没有必要的列")
        # unuse_col=['item_city','user_city','device','author_id','music_id',]  #'uid','item_id'这两列不能删除，后面提交结果的时候应该要用到
        # df_train=self.dropUnuseCols(df_train,unuse_col)
        # df_test=self.dropUnuseCols(df_test,unuse_col)

        print("表中含有为null的字段，主要产生在leftjoin的时候")
        # df_train=df_train.na.fill({'uid_author_id_count_bin':1,'uid_author_id_count_ratio':0,\
        #                            'uid_item_city_count_bin':1,'uid_item_city_count_ratio':0,\
        #                            'uid_channel_count_bin':1,'uid_channel_count_ratio':0,\
        #                            'uid_music_id_count_bin':1,'uid_music_id_count_ratio':0,\
        #                            'uid_device_count_bin':1,'uid_device_count_ratio':0,\
        #                            'author_id_channel_count_bin':1,'author_id_channel_count_ratio':0,\
        #                            'author_id_user_city_count_bin':1,'author_id_user_city_count_ratio':0,\
        #                            'author_id_item_city_count_bin':1,'author_id_item_city_count_ratio':0,\
        #                            'author_id_music_id_count_bin':1,'author_id_music_id_count_ratio':0,\
        #                            'uid_channel_device_count_bin':1,'uid_channel_device_count_ratio':0,\
        #                            'author_id_item_city_music_id_bin':1,'author_id_item_city_music_id_ratio':0
        #                            })
        df_train=df_train.na.fill({'user_city_count_bin':1,'user_city_count_ratio':0})
        #user_city_count_bin,device_count_bin  这两个是step1_single中漏掉的两个字段
        df_test=df_test.na.fill({'user_city_count_bin':1,'user_city_count_ratio':0,\
                                 'device_count_bin':-1,'device_count_ratio':0,\
                                   'uid_author_id_count_bin':1,'uid_author_id_count_ratio':0,\
                                   'uid_item_city_count_bin':1,'uid_item_city_count_ratio':0,\
                                   'uid_channel_count_bin':1,'uid_channel_count_ratio':0,\
                                   'uid_music_id_count_bin':1,'uid_music_id_count_ratio':0,\
                                   'uid_device_count_bin':1,'uid_device_count_ratio':0,\
                                   'author_id_channel_count_bin':1,'author_id_channel_count_ratio':0,\
                                   'author_id_user_city_count_bin':1,'author_id_user_city_count_ratio':0,\
                                   'author_id_item_city_count_bin':1,'author_id_item_city_count_ratio':0,\
                                   'author_id_music_id_count_bin':1,'author_id_music_id_count_ratio':0,\
                                   'uid_channel_device_count_bin':1,'uid_channel_device_count_ratio':0,\
                                   'author_id_item_city_music_id_count_bin':1,'author_id_item_city_music_id_count_ratio':0
                                   })

        print("查看test缺失值")
        df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()
        print("查看train缺失值")  #以防万一，可能会漏掉哪个字段
        df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()

        print('-------5.保存数据预处理结果-------')
        test_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_test_single_cross'
        os.system("hadoop fs -rm -r {}".format(test_file_path))
        df_test.rdd.map(tuple).saveAsPickleFile(test_file_path)

        del df_test
        gc.collect()

        train_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_train_single_cross'
        os.system("hadoop fs -rm -r {}".format(train_file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train.rdd.map(tuple).saveAsPickleFile(train_file_path)


if __name__ == "__main__":
    spark_job = SparkFEProcess()
    df_train,df_test=spark_job.data_describe()
    spark_job.data_explore(df_train,df_test)

