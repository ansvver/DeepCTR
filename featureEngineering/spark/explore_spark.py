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
        print('starto read data for rdd:')
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
        print("duration_time应该根据喜欢和不喜欢来分箱")
        print("查看duration_time的分布")
        print()
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

        print('--------2、统计特征：count、ratio、nunique、ctr相关特征')
        print("计算基础特征和交叉特征的count、类别偏好的ratio")
        count_feats_list = []

        print('single feature count')
        count_feats_list.extend([[c] for c in df_train.columns if c not in ['time', 'channel', 'like', 'finish','dutration_time',"time_day","item_pub_month","item_pub_day","item_pub_hour","item_pub_minute"]])
        print(count_feats_list)

        print('cross count')
        users = ['uid']
        authors = ['item_id', 'user_city', 'author_id', 'item_city', 'channel', 'music_id', 'device','item_pub_hour']
        count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

        users = ['author_id']
        authors = ['channel', 'user_city', 'item_city', 'music_id',  'item_pub_hour']
        count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

        count_feats_list.append(['uid', 'user_city', 'channel', 'device'])
        count_feats_list.append(['author_id', 'item_city', 'music_id','item_pub_hour'])
        print("计算count的字段有以下这些")
        print(count_feats_list)

        for i in range(len(count_feats_list)):
           group_cols=count_feats_list[i]
           new_feature = '_'.join(group_cols)
           #判断是几维交叉特征，并进行拼接，再计算每个特征值的个数count,并完成映射
           if len(group_cols)==1:
              if new_feature in ["music_id"] :
                  df1 = df_train.where(df_train[new_feature]!=-1).groupby(new_feature).count()\
                          .withColumnRenamed('count',new_feature+'_count')
              else:
                  df1 = df_train.groupby(new_feature).count()\
                          .withColumnRenamed('count',new_feature+'_count')
              #类别偏好的ratio比例
              count_min = df1.select(fn.min(df1[new_feature+'_count'])).collect()[0][0]
              count_max = df1.select(fn.max(df1[new_feature+'_count'])).collect()[0][0]
              # F.bround("Rank", scale=4)
              df1=df1.withColumn(new_feature+'_count_ratio', fn.bround(((df1[new_feature+'_count']-fn.lit(count_min)) /((fn.lit(count_max)-fn.lit(count_min)).cast(typ.IntegerType()))),scale=3))
              # print("查看df1_1")
              # df1.show(5,truncate=False)
              if new_feature=="device":   #[1.0, 16.0, 46.0, 102.0, 204.0, 410.0, 10389.0] 修改
                 percent_list=[0,10,20,30,40,50,60,70,80,90,100]
              elif new_feature=="author_id":  #[1.0, 2.0, 7.0, 32.0, 78.0, 276186.0]
                  percent_list=[0,50,75,90,95,100]
              elif new_feature=="music_id":   #[1.0, 3.0, 13.0, 73.0, 211.0, 193640.0]
                 percent_list=[0,50,75,90,95,100]   #每个percent_list不相同
              elif new_feature=="uid":       #分箱[1.0, 104.0, 329.0, 741.0, 1131.0, 10389.0]
                  percent_list=[0,50,75,90,95,100]
              elif new_feature=="item_id":   #[1.0, 1.0, 2.0, 7.0, 14.0, 6911.0]  分箱修改
                  percent_list=[0,75,90,95,100]
              elif new_feature=="user_city":  #[1.0, 21935.5, 54519.5, 110179.0, 146319.75, 3789087.0] 修改
                  percent_list=[0,10,20,30,40,50,60,70,80,90,100]
              elif new_feature=="item_city":  #[1.0, 14725.0, 48576.0, 122887.0, 206845.5, 744265.0]  修改
                  percent_list=[0,10,20,30,40,50,60,70,80,90,100]
              else:
                  percent_list=[0,10,20,30,40,50,60,70,80,90,100]

              df1=self.bining(sqlContext,df1,new_feature+'_count',percent_list)
              # print(df1.show(5,truncate=False))
              df_train=df_train.join(df1,new_feature,'left')
              # print("train")
              # df_train.show(5,truncate=False)   #ratio是一个连续变量，范围0-1
              df_test=df_test.join(df1,new_feature,'left')
              # print("test")
              # df_test.show(5,truncate=False)   #ratio是一个连续变量，范围0-1
              del df1
              gc.collect()
           print("输出所有一维特征处理后的结果")
           df_train.show(1,truncate=False)
           df_train.printSchema()
           df_test.show(1,truncate=False)
           df_train.printSchema()

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
              # print("查看df1_1")
              # df2.show(5,truncate=False)
              if new_feature=="uid_item_id":
                 percent_list=[0,20,35,50,65,85,100]   #每个percent_list不相同
              else:
                 percent_list=[0,50,75,90,95,100]
              # elif new_feature=="uid_user_city":
              #     percent_list=[0,50,75,90,95,100]
              # elif new_feature=="uid_author_id":
              #    percent_list=[0,50,75,90,95,100]   #每个percent_list不相同
              # elif new_feature=="uid_item_city":
              #     percent_list=[0,50,75,90,95,100]
              # elif new_feature=="uid_channel":
              #     percent_list=[0,50,75,90,95,100]
              # elif new_feature=="uid_music_id":
              #     percent_list=[0,50,75,90,95,100]
              # elif new_feature=="uid_device":
              #     percent_list=[0,50,75,90,95,100]
              # elif new_feature=="uid_time_pub_hour":
              #     percent_list=[0,50,75,90,95,100]

              # ['uid', 'item_id'], ['uid', 'user_city'], ['uid', 'author_id'], ['uid', 'item_city'], ['uid', 'channel'], ['uid', 'music_id'],
              #  ['uid', 'device'], ['uid', 'time_pub_hour']
              #['author_id', 'channel'], ['author_id', 'user_city'], ['author_id', 'item_city'], ['author_id', 'music_id'], ['author_id', 'time_pub_hour']

              df2=self.bining(sqlContext,df2,new_feature+'_count',percent_list)
              print("查看df2_2")
              df2.show(5,truncate=False)
              df_train=df_train.join(df2,new_feature,'left')
              # print("train")
              # df_train.show(5,truncate=False)   #ratio是一个连续变量，范围0-1
              df_test=df_test.join(df2,new_feature,'left')
              # print("test")
              # df_test.show(5,truncate=False)


           if len(group_cols)==4:
              print("开始处理4维交叉变量")
              df_train=df_train.withColumn(new_feature, fn.concat_ws('_',df_train[group_cols[0]].cast(typ.StringType()),df_train[group_cols[1]].cast(typ.StringType()),
                                                             df_train[group_cols[2]].cast(typ.StringType()),df_train[group_cols[3]].cast(typ.StringType()))
                                                           )
              df_test=df_test.withColumn(new_feature, fn.concat_ws('_',df_test[group_cols[0]].cast(typ.StringType()),df_test[group_cols[1]].cast(typ.StringType()),
                                                             df_test[group_cols[2]].cast(typ.StringType()),df_test[group_cols[3]].cast(typ.StringType()))
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
              df3.show(5,truncate=False)
              df_train=df_train.join(df3,new_feature,'left')
              # print("train")
              # df_train.show(5,truncate=False)
              # ['uid', 'user_city', 'channel', 'device'], ['author_id', 'item_city', 'music_id', 'time_pub_hour']
              df_test=df_test.join(df3,new_feature,'left')
              # print("test")
              # df_test.show(5,truncate=False)
        # df.show(5,truncate=False)
        print("删除没有必要的列")
        unuse_col=['item_city','user_city','device','author_id','music_id',]  #'uid','item_id'这两列不能删除，后面提交结果的时候应该要用到
        df_train=self.dropUnuseCols(df_train,unuse_col)
        df_test=self.dropUnuseCols(df_test,unuse_col)

        print("表中含有为null的字段，主要产生在leftjoin的时候")
        print("这一步先不做，三表联合的时候会填充")
        # df_train=df_train.na.fill(-1)
        # df_test=df_test.na.fill(-1)

        print("查看train的统计信息")
        desc = df_train.describe()
        desc.show()
        print("查看test的统计信息")
        desc = df_test.describe()
        desc.show()


        print('-------5.保存数据预处理结果-------')
        test_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_test_new'
        os.system("hadoop fs -rm -r {}".format(test_file_path))
        df_test.rdd.map(tuple).saveAsPickleFile(test_file_path)

        del df_test
        gc.collect()

        train_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_train_new'
        os.system("hadoop fs -rm -r {}".format(train_file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train.rdd.map(tuple).saveAsPickleFile(train_file_path)




        '''
        #device出现的次数
        dfDeviceCount = df.groupby('device').count() \
             .withColumnRenamed("count", "device_Cnt")
        print('device出现次数')
        # dfDeviceCount.show(10)
        percent_list=[0,20,35,50,65,80,100]
        dfDeviceCount_spark=self.bining(sqlContext,dfDeviceCount,"device_Cnt",percent_list)

        #author_id出现的次数
        dfAuthoridCount = df.groupby('author_id').count() \
             .withColumnRenamed("count", "authorid_Cnt")
        print('authorid出现次数')
        # dfAuthoridCount.show(10)
        # dfAuthoridCount.describe().show()
        percent_list=[0,50,75,90,95,100]
        dfAuthoridCount_spark=self.bining(sqlContext,dfAuthoridCount,"authorid_Cnt",percent_list)

        #music_id出现的次数
        dfMusicidCount = df.where(df['music_id']!=-1).groupby('music_id').count() \
             .withColumnRenamed("count", "musicid_Cnt")
        print('musicic出现次数')
        # dfMusicidCount.show(10)
        # dfMusicidCount.show(10)
        percent_list=[0,50,75,90,95,100]
        dfMusicidCount_spark=self.bining(sqlContext,dfMusicidCount,"musicid_Cnt",percent_list)


        # 一个用户观看的视频数
        dfCntUserPlayVideo = df.groupby('uid') \
            .agg({"item_id": "count"}) \
            .withColumnRenamed("count(item_id)", "uid_playCnt")
        print('一个用户观看的视频数')
        # dfCntUserPlayVideo.show(10)
        percent_list=[0,20,35,50,65,85,100]
        dfCntUserPlayVideo_spark=self.bining(sqlContext,dfCntUserPlayVideo,"uid_playCnt",percent_list)

        #每个视频被多少用户观看
        dfCntvidPlayVideo = df.groupby('item_id') \
            .agg({"uid": "count"}) \
            .withColumnRenamed("count(uid)", "itemid_playCnt")
        print('每个视频被多少用户观看')
        # dfCntvidPlayVideo.show(10)
        percent_list=[50,65,85,95,100]
        dfCntvidPlayVideo_spark=self.bining(sqlContext,dfCntvidPlayVideo,"itemid_playCnt",percent_list)
        ##两列对city的处理是针对全部数据的处理，正好处理不妥当
        print('对user_city进行统计，根据finish、like的情况，划分城市等级，后对测试集相关字段进行映射')
        print('为item_city分组求finish、like列均值，再将两列的值按比例求和')

        df_user_city_score=self.city_col_deal(df,'user_city')
        percent_list=[0,5,50,95,100]
        dfUserCityScore_spark=self.bining(sqlContext,df_user_city_score,"user_city_score",percent_list)
        # dfUserCityScore_spark.show()

        df_item_city_score=self.city_col_deal(df,'item_city')
        percent_list=[0,10,30,70,90,100]
        dfItemCityScore_spark=self.bining(sqlContext,df_item_city_score,"item_city_score",percent_list)
        # dfItemCityScore_spark.show()




        df_bin=df.join(dfDeviceCount_spark,'device','left') \
                    .join(dfAuthoridCount_spark,'author_id','left') \
                    .join(dfMusicidCount_spark,'music_id','left') \
                    .join(dfCntUserPlayVideo_spark,'uid','left') \
                    .join(dfCntvidPlayVideo_spark,'item_id','left') \
                    .join(dfUserCityScore_spark,'user_city','left') \
                    .join(dfItemCityScore_spark,'item_city','left')
        # .join(df_item_hour,'item_id','left') \
        #删除没有必要的列
        unuse_col=['item_city','user_city','device','author_id','music_id',]  #'uid','item_id'这两列不能删除，后面提交结果的时候应该要用到
        df_bin=self.dropUnuseCols(df_bin,unuse_col)

        # df_train_bin.show(10)
        # 给表中为null的字段填充为-1，例如，给musicid_Cnt_bin中的null，填充为-1
        df_bin=df_bin.na.fill(-1)
        # df_train_bin.show(truncate=False)
        # desc = df_train_bin.describe()
        # desc.show()
        # print('返回各列名和数据类型')
        # print(df_train_bin.dtypes)
        #根据train_count拆分训练集和测试集

        #选出某些列为-1的值
        df_train_bin=df_bin.filter(df_bin['like']!=-1).filter(df_bin['finish']!=-1)
        print("训练集的数量")
        print(df_train_bin.count())

        df_test_bin=df_bin.filter(df_bin['like']==-1).filter(df_bin['finish']==-1)
        print("测试集的数量")
        print(df_test_bin.count())

        #对测试集进行分箱处理
        #item_id  item_hour 不需要进行join  根据自身运算得到即可
        #将 unix 格式的时间戳转换为指定格式的日期,提取小时

        print('-------5.保存数据预处理结果-------')
        test_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_test_bin'
        os.system("hadoop fs -rm -r {}".format(test_file_path))
        df_test_bin.rdd.map(tuple).saveAsPickleFile(test_file_path)

        del df_test_bin
        gc.collect()

        train_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_train_bin'
        os.system("hadoop fs -rm -r {}".format(train_file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train_bin.rdd.map(tuple).saveAsPickleFile(train_file_path)

        '''



if __name__ == "__main__":
    spark_job = SparkFEProcess()

    df_train,df_test=spark_job.data_describe()

    spark_job.data_explore(df_train,df_test)