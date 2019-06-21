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

        sparkConf = SparkConf().setAppName("feature engineering on spark of explore_spark") \
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
        print('start to read data for rdd:')
        rawRdd_train = self.read_rdd('final_track2_train.txt').map(lambda line : line.split('\t'))
        rawRdd_test = self.read_rdd('final_track2_test_no_anwser.txt').map(lambda line : line.split('\t'))
        print('finish read rdd, start to init action log rdd:')
        actionLogRdd_train = rawRdd_train.map(
            lambda x :(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
                       int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))
        # total = actionLogRdd_train.count()
        # print('total: ' + str(total))

        actionLogRdd_test = rawRdd_test.map(
            lambda x :(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
                       int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))

        #转化为dataframe
        sqlContext = SQLContext(self.sc)
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

        dfactionLog_train = sqlContext.createDataFrame(actionLogRdd_train, actionLogSchema)
        dfactionLog_test = sqlContext.createDataFrame(actionLogRdd_test, actionLogSchema)

        dfactionLog_train=dfactionLog_train.filter(dfactionLog_train['duration_time']<=300)
        dfactionLog_test=dfactionLog_test.filter(dfactionLog_test['duration_time']<=300)
        #train和test合并，并且保存保存train的数量，以便拆分   union可能会改变frame中的顺序
        # df=dfactionLog_train.union(dfactionLog_test)
        # train_count=dfactionLog_train.count()
        # print("训练集的数量"+str(train_count))
        # test_count=dfactionLog_test.count()
        # print("测试集的数量"+str(test_count))

        # print('-------2.finish\like各特征下值的个数-------------')
        # df.agg( fn.countDistinct('finish').alias('finish_distinct'), \
        #         fn.countDistinct('like').alias('like_distinct')
        #         ).show()
        # print("各特征下的最大值,最小值")
        # df.describe().show()


        return dfactionLog_train, dfactionLog_test

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
        # print(pandas_df)

        #修改pandas中的列名
        pandas_df.rename(columns={col:col+'_bin'}, inplace = True)
        df_spark= sqlContext.createDataFrame(pandas_df)
        # df_spark.show()
        return  df_spark


    def city_col_deal(self,df,col):
        df_city_score=df.groupBy(col).avg('finish', 'like') \
            .withColumnRenamed("avg(finish)","avg_finish").withColumnRenamed("avg(like)","avg_like")
        df_city_score=df_city_score.withColumn(col+'_score', df_city_score.avg_finish*0.7+df_city_score.avg_like*0.3)\
                              .select(col,fn.bround(col+'_score', scale=4).alias(col+'_score'))
        return df_city_score

    def dropUnuseCols(self,df,unuse_col):
        '''
        #删除没有必要的列
        #device|time|author_id|music_id| uid|item_id|
        #保留一下列
        #  user_city|item_city|channel|finish|like|duration_time
        # device_Cnt_bin|item_pub_hour||authorid_Cnt_bin|musicid_Cnt_bin|uid_playCnt_bin|itemid_playCnt_bin
        '''
        # unuse_col=['device','time','author_id','music_id','uid','item_id']
        for col in unuse_col:
            df=df.drop(col)
        return df


    def data_explore(self,df_train,df_test):

        sqlContext = SQLContext(self.sc)
        print("把duration_time当作连续变量")

        print("------------1、通过时间戳获取年月日时分，(没有工作日特征，月日交叉表示节日特征,年份转化有问题)-----------------")

        #作品发布时间-作品发布的最早时间,转化为day
        time_min = df_train.select(fn.min(df_train['time'])).collect()
        df_train=df_train.withColumn('time_day', ((df_train.time-fn.lit(time_min[0][0])) /fn.lit(3600 * 24)).cast(typ.IntegerType()))
        # df_train=df_train.withColumn('time_strDate',fn.from_unixtime(df_train.time , "yyyy-MM-dd HH:mm:ss"))
        #将 unix 格式的时间戳转换为指定格式的日期,提取小时
        df_train=df_train.withColumn('item_pub_month',fn.from_unixtime(df_train.time , "M").cast(typ.IntegerType()))
        df_train=df_train.withColumn('item_pub_day',fn.from_unixtime(df_train.time , "d").cast(typ.IntegerType()))
        df_train=df_train.withColumn('item_pub_hour',fn.from_unixtime(df_train.time , "k").cast(typ.IntegerType()))
        df_train=df_train.withColumn('item_pub_minute',fn.from_unixtime(df_train.time , "m").cast(typ.IntegerType()))
        print("查看month,day,hour,minute的提取是否正确")
        df_train.show(truncate=False)
        df_train=df_train.drop('time')
        #对时间提取的这部分字段进行count后进行分箱并不明显，就直接当作类别变量处理就可以了，另外增加pos_neg_ratio特征



        df_test=df_test.withColumn('time_day', ((df_test.time-fn.lit(time_min[0][0])) /fn.lit(3600 * 24)).cast(typ.IntegerType()))
        df_test=df_test.withColumn('item_pub_month',fn.from_unixtime(df_test.time , "M").cast(typ.IntegerType()))
        df_test=df_test.withColumn('item_pub_day',fn.from_unixtime(df_test.time , "d").cast(typ.IntegerType()))
        df_test=df_test.withColumn('item_pub_hour',fn.from_unixtime(df_test.time , "k").cast(typ.IntegerType()))
        df_test=df_test.withColumn('item_pub_minute',fn.from_unixtime(df_test.time , "m").cast(typ.IntegerType()))
        df_test=df_test.drop('time')

        print('--------2、对duration_time 和time_day  进行处理  count、sum_like 、sum_finish')




if __name__ == "__main__":
    spark_job = SparkFEProcess()

    df_train,df_test=spark_job.data_describe()

    spark_job.data_explore(df_train,df_test)
