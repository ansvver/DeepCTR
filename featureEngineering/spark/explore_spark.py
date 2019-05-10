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

import numpy as np
import gc

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


        return dfactionLog_train,dfactionLog_test

    def bining(self,sqlContext,df,col,percent_list):
        # 不想给定分类标准，可以让spark自动给我们分箱
        #//和Bucketizer类似：将连续数值特征转换离散化。但这里不再自己定义splits（分类标准），而是定义分几箱就可以了。
        # val quantile = new QuantileDiscretizer().setNumBuckets(5).setInputCol("age").setOutputCol("quantile_feature")
        # //因为事先不知道分桶依据，所以要先fit,相当于对数据进行扫描一遍，取出分位数来，再transform进行转化。
        # val quantiledf = quantile.fit(df).transform(df)
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
        # print(col+'查看分箱')
        # print(bins)
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

        print('-------3.观察训练集和测试集的数据差异-------------')
        sqlContext = SQLContext(self.sc)
        #对类别变量进行处理
        #查看在测试集中出现的值，但在训练集中没有出现，对于uid来说，则是新用户，对于item_id来说，则是新item
        # ca_col=['item_city']
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
        item_city: 5  测试集中有5个城市，在训练集中没有出现，后期训练集映射的时候，将这5个城市划分为一类，即小众城市
        channel:
        music_id:
        device:
       '''
        #通过数据探索，发现duration_time中70000秒是异常数据，
        #查看duration_time=0，duration_time>5*60的记录数，抖音视频时长范围为(0,300]
        '''
        print('检查duration_time中异常数据')
        print('300+视频，看下用户对它的一些点击行为情况')
        df_train.filter(df_train['duration_time']>300).show(223)
        print(df_train.filter(df_train['duration_time']>300).select('item_id').distinct().count())
        df_train.filter(df_train['duration_time']>300).select('item_id').distinct().show(223)
        print(df_train.where(df_train['duration_time']==0).count())   #224
        print(df_train.where(df_train['duration_time']>300).count())  #223
        print(df_train.filter("duration_time > 0 and duration_time <=60" ).count())  #19621510
        print(df_train.filter("duration_time > 60 and duration_time <=300" ).count())  #383
        '''

        #将 unix 格式的时间戳转换为指定格式的日期,提取小时
        df_train=df_train.withColumn('item_pub_hour',fn.from_unixtime(df_train.time , "h")).cast(typ.IntegerType())

        #time列在这里就可以删掉了  cast(typ.IntegerType)
        df_train=df_train.drop('time')
        # df_train.show(truncate=False)

        print('对user_city进行统计，根据finish、like的情况，划分城市等级，后对测试集相关字段进行映射')
        print('为item_city分组求finish、like列均值，再将两列的值按比例求和')

        df_user_city_score=self.city_col_deal(df_train,'user_city')
        percent_list=[0,5,50,95,100]
        dfUserCityScore_spark=self.bining(sqlContext,df_user_city_score,"user_city_score",percent_list)
        # dfUserCityScore_spark.show()

        df_item_city_score=self.city_col_deal(df_train,'item_city')
        percent_list=[0,10,30,70,90,100]
        dfItemCityScore_spark=self.bining(sqlContext,df_item_city_score,"item_city_score",percent_list)
        # dfItemCityScore_spark.show()




        print('-------4.统计并汇总用户行为数据-------')

        #device出现的次数
        dfDeviceCount = df_train.groupby('device').count() \
             .withColumnRenamed("count", "device_Cnt")
        print('device出现次数')
        # dfDeviceCount.show(10)
        percent_list=[0,20,35,50,65,80,100]
        dfDeviceCount_spark=self.bining(sqlContext,dfDeviceCount,"device_Cnt",percent_list)

        #author_id出现的次数
        dfAuthoridCount = df_train.groupby('author_id').count() \
             .withColumnRenamed("count", "authorid_Cnt")
        print('authorid出现次数')
        # dfAuthoridCount.show(10)
        # dfAuthoridCount.describe().show()
        percent_list=[0,50,75,90,95,100]
        dfAuthoridCount_spark=self.bining(sqlContext,dfAuthoridCount,"authorid_Cnt",percent_list)

        #music_id出现的次数
        dfMusicidCount = df_train.where(df_train['music_id']!=-1).groupby('music_id').count() \
             .withColumnRenamed("count", "musicid_Cnt")
        print('musicic出现次数')
        # dfMusicidCount.show(10)
        percent_list=[0,50,75,90,95,100]
        dfMusicidCount_spark=self.bining(sqlContext,dfMusicidCount,"musicid_Cnt",percent_list)


        # 一个用户观看的视频数
        dfCntUserPlayVideo = df_train.groupby('uid') \
            .agg({"item_id": "count"}) \
            .withColumnRenamed("count(item_id)", "uid_playCnt")
        print('一个用户观看的视频数')
        # dfCntUserPlayVideo.show(10)
        percent_list=[0,20,35,50,65,85,100]
        dfCntUserPlayVideo_spark=self.bining(sqlContext,dfCntUserPlayVideo,"uid_playCnt",percent_list)

        #每个视频被多少用户观看
        dfCntvidPlayVideo = df_train.groupby('item_id') \
            .agg({"uid": "count"}) \
            .withColumnRenamed("count(uid)", "itemid_playCnt")
        print('每个视频被多少用户观看')
        # dfCntvidPlayVideo.show(10)
        percent_list=[50,65,85,95,100]
        dfCntvidPlayVideo_spark=self.bining(sqlContext,dfCntvidPlayVideo,"itemid_playCnt",percent_list)

        df_train_bin=df_train.join(dfDeviceCount_spark,'device','left') \
                    .join(dfAuthoridCount_spark,'author_id','left') \
                    .join(dfMusicidCount_spark,'music_id','left') \
                    .join(dfCntUserPlayVideo_spark,'uid','left') \
                    .join(dfCntvidPlayVideo_spark,'item_id','left') \
                    .join(dfUserCityScore_spark,'user_city','left') \
                    .join(dfItemCityScore_spark,'item_city','left')
        # .join(df_item_hour,'item_id','left') \
        #删除没有必要的列
        unuse_col=['item_city','user_city','device','author_id','music_id',]  #'uid','item_id'这两列不能删除，后面提交结果的时候应该要用到
        df_train_bin=self.dropUnuseCols(df_train_bin,unuse_col)

        # df_train_bin.show(10)
        # 给表中为null的字段填充为-1，例如，给musicid_Cnt_bin中的null，填充为-1
        df_train_bin=df_train_bin.na.fill(-1)
        # df_train_bin.show(truncate=False)
        # desc = df_train_bin.describe()
        # desc.show()
        # print('返回各列名和数据类型')
        # print(df_train_bin.dtypes)


        #对测试集进行分箱处理
        #item_id  item_hour 不需要进行join  根据自身运算得到即可
        #将 unix 格式的时间戳转换为指定格式的日期,提取小时,#将string类转化为int类型
        df_test=df_test.withColumn('item_pub_hour', fn.from_unixtime(df_test.time , "h")).cast(typ.IntegerType())

        df_test=df_test.drop('time')
        df_test_bin=df_test.join(dfDeviceCount_spark,'device','left') \
                    .join(dfAuthoridCount_spark,'author_id','left') \
                    .join(dfMusicidCount_spark,'music_id','left') \
                    .join(dfCntUserPlayVideo_spark,'uid','left') \
                    .join(dfCntvidPlayVideo_spark,'item_id','left') \
                    .join(dfUserCityScore_spark,'user_city','left') \
                    .join(dfItemCityScore_spark,'item_city','left')

        # .join(df_item_hour,'item_id','left') \
        #删除没有必要的列
        df_test_bin=self.dropUnuseCols(df_test_bin,unuse_col)
        df_test_bin=df_test_bin.na.fill(-1)
        df_test_bin.show(truncate=False)
        # desc = df_train_bin.describe()
        # desc.show()
        # print('返回各列名和数据类型')
        # print(df_train_bin.dtypes)

        #清理内存
        del dfDeviceCount_spark
        del dfAuthoridCount_spark
        del dfMusicidCount_spark
        del dfCntUserPlayVideo_spark
        del dfCntvidPlayVideo_spark
        del dfUserCityScore_spark
        del dfItemCityScore_spark
        gc.collect()

        #df_train_bin,df_test_bin保存后读取
        # item_id,uid,music_id,author_id,device,user_city,item_city,channel,finish,like,time,duration_time,uid_Cnt,,device_Cnt,
        # authorid_Cnt,musicid_Cnt,uid_playCnt,itemid_playCnt,item_pub_hour

        print('-------5.保存数据预处理结果-------')
        test_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_test_bin'
        os.system("hadoop fs -rm -r {}".format(test_file_path))
        df_test_bin.rdd.map(tuple).saveAsPickleFile(test_file_path)

        del df_test_bin
        gc.collect()

        train_file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'actLog_train_bin'
        os.system("hadoop fs -rm -r {}".format(train_file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train_bin.rdd.map(tuple).saveAsPickleFile(train_file_path)


        #
        print('数据保存结束')

        #用以下方式读取
        # rdd = self.sc.pickleFile(test_file_path)
        # rdd.collect()
        # new_df = sqlContext.createDataFrame(rdd, chema)
        # new_df.show()


        # return df_train_bin,df_test_bin

        # return df_train_bin,df_test_bin

        #保存dfAll,转化为rdd.collect(),一条条保存到txt文件，以\t分割
        # 数据写到hdfs，而且以csv格式保存
        # dfAll.write.mode("overwrite").options(header="true").csv("/home/ai/da/da_aipurchase_dailysale_for_ema_predict.csv")

        #显示csv有问题
        #显示csv有问题
        # df_user_city_score.write.csv(path="/user/hadoop/icmechallenge2019/track2/data/user_city_score2.csv", header="true", mode="overwrite", sep="\t")
        #显示save有问题
        # df_user_city_score.write.format('csv').mode('overwrite').option("header", "true").save("/user/hadoop/icmechallenge2019/track2/data/user_city_score1.csv")

        #保存csv文件
        # def toCSVLine(data):
        #   return ','.join(str(d) for d in data)
        #
        # lines = labelsAndPredictions.map(toCSVLine)
        # lines.saveAsTextFile('hdfs://my-node:9000/tmp/labels-and-predictions.csv')


        # ca_col_before = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
        #                'music_id', 'device',]
        # co_col = ['duration_time']  # track2_time作品发布时间，作品时长
        # #各变量与y的相关性
        # all_col=ca_col_before+co_col
        # for i in all_col :
        #     for j in label:
        #         print('{}和{}的相关性:'.format(i,j),df_train.corr(i,j))
        #
        #
        # #预处理后输入模型的变量
        # ca_col_after= ['uid_playCnt', 'user_city', 'itemid_playCnt', 'authorid_Cnt', 'item_city', 'channel',
        #                'musicid_Cnt', 'device_Cnt', 'item_pub_hour',]
        # all_col_after=ca_col_after+co_col
        # for i in all_col_after :
        #     for j in label:
        #         print('{}和{}的相关性:'.format(i,j),dfAll.corr(i,j))



if __name__ == "__main__":
    spark_job = SparkFEProcess()

    dfactionLog_train,dfactionLog_test=spark_job.data_describe()

    spark_job.data_explore(dfactionLog_train,dfactionLog_test)