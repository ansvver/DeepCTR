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
# from featureEngineering.spark.preprocess_functions import *
# import preprocess_functions
import datetime
from pyspark.sql.functions import udf

# workspace_path='/data/code/DeepCTR/'
# hdfs_data_path = '/user/hadoop/icmechallenge2019/track2/data/'

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


        return dfactionLog_train,dfactionLog_test


    def data_explore(self,df_train,df_test):
        # print('-------1.统计并汇总用户行为数据-------')
        # desc = df.describe()
        # desc.show()
        #
        # print('-------2.各特征下值的个数-------------')
        # df.agg(fn.countDistinct('uid').alias('uid_distinct'), \
        #                 fn.countDistinct('user_city').alias('user_city_distinct'), \
        #                 fn.countDistinct('item_id').alias('item_id_distinct'), \
        #                 fn.countDistinct('author_id').alias('author_id_distinct'), \
        #                 fn.countDistinct('item_city').alias('item_city_distinct'), \
        #                 fn.countDistinct('channel').alias('channel_distinct'), \
        #                 fn.countDistinct('finish').alias('finish_distinct'), \
        #                 fn.countDistinct('like').alias('like_distinct'), \
        #                 fn.countDistinct('music_id').alias('music_id_distinct'), \
        #                 fn.countDistinct('device').alias('device_distinct')
        #                 ).show()
        # 至此花费5分钟时间
        '''
        训练集中总记录数：19622340
        各特征下值的个数如下：
        uid_distinct:    70711
        user_city_distinct:  396
        item_id_distinct:  3687157
        author_id_distinct:  778113
        item_city_distinct:  456
        channel_distinct:    5
        finish_distinct:     2
        like_distinct:       2
        music_id_distinct:   82840
        device_distinct:     71681
        '''
        print('-------3.观察训练集和测试集的数据差异-------------')
        ca_col = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'device']
        co_col = ['time', 'duration_time']  # track2_time作品发布时间，作品时长
        #pysparksql中将time时间转化为年月日的形式

        timestamps = df_train.rdd.map(lambda fields: fields[10])
        day = timestamps.map(lambda ts: datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"))
        print(day.take(5))
        #datetime.datetime(3652, 3, 24, 13, 21, 39)
        # [13, 22, 22, 23, 15]  .hour获取时间

        label=['finish','like']
        #对类别变量进行处理
        #查看在测试集中出现的值，但在训练集中没有出现，对于uid来说，则是新用户，对于item_id来说，则是新item
        # for col in ca_col:
        #     us_train=df_train.select(col).distinct().rdd.map(lambda r: r[0]).collect()
        #     us_test=df_test.select(col).distinct().rdd.map(lambda r: r[0]).collect()
        #     ret = [ i for i in us_test if i not in us_train ]
        #     print(col,len(ret))
        '''
        uid: 3263
        user_city: 0
        item_id:
        author_id:
        item_city:
        channel:
        music_id:
        device:
       '''
        print('-------4.统计并汇总用户行为数据-------')
        print('返回各列名和数据类型')
        print( df_train.dtypes)
        print(df_train.show())
        #uid出现的次数
        dfUserCount = df_train.groupby('uid').count() \
             .withColumnRenamed("count", "uid_Cnt")
        print('uid出现次数')
        dfUserCount.show(10)

        # 用户与视频的情况
        # 特别注意使用agg后，对应列的顺序可能会发生变化，所以toDF一定要对应好
        dfRecordUserToVideo = df_train.groupby(['uid', 'item_id']) \
            .agg({ "finish": "sum","like": "sum"}) \
            .toDF('uid', 'item_id', 'like', 'finish')
        print('一个用户对某个item的点赞、完成情况统计')
        dfRecordUserToVideo.show(10)

        # 一个用户观看的视频数
        dfCntUserPlayVideo = df_train.groupby('uid') \
            .agg({"item_id": "count"}) \
            .withColumnRenamed("count(item_id)", "uid_playCnt")
        print('一个用户观看的视频数')
        print(dfCntUserPlayVideo.show(10))

        # 每个用户的互动情况
        dfBehavePerUser = df_train.groupBy(['uid']) \
            .sum('finish', 'like', ) \
            .toDF('uid', 'uFinishCnt', 'uLikeCnt')
        print('每个用户like、finish的情况')
        print(dfBehavePerUser.show(10))

        #每个视频被多少用户观看
        dfCntvidPlayVideo = df_train.groupby('item_id') \
            .agg({"uid": "count"}) \
            .withColumnRenamed("count(uid)", "itemid_playCnt")
        print('每个视频被多少用户观看')
        print(dfCntvidPlayVideo.show(10))

        # 每个视频被互动情况
        dfBehavedPerVideo = df_train.groupBy(['item_id']) \
            .sum('finish', 'like') \
            .toDF('item_id', 'vFinishCnt', 'vLikedCnt')
        print('每个视频被like、finish的情况')
        print(dfBehavedPerVideo.show(10))


        #各变量与y的相关性
        all_col=ca_col+co_col
        for i in all_col :
            for j in label:
                print('{}和{}的相关性'.format(i,j))
                print(df_train.corr(i,j))
        #将time转化为年月日时分秒的形式，截取时，变成类别变量



















if __name__ == "__main__":
    spark_job = SparkFEProcess()

    dfactionLog_train,dfactionLog_test=spark_job.data_describe()

    spark_job.data_explore(dfactionLog_train,dfactionLog_test)