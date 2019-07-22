from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import  configparser
import pyspark.sql.types as typ
import os
import gc
import pyspark.sql.functions as fn
from pyspark.sql.functions import udf
# import pyspark.sql.functions  as psf
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import StandardScaler, VectorAssembler
import pickle

class SparkFEProcess:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark of concate") \
            .set("spark.ui.showConsoleProgress", "false") \
            .set("spark.driver.maxResultSize", "4g") \
            .set("set spark.sql.execution.arrow.enabled","true")
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

    def dropUnuseCols(self,df,unuse_col):
        for col in unuse_col:
            df=df.drop(col)
        return df

    def data_describe(self):
        sqlContext = SQLContext(self.sc)
        rootPath=self.parser.get("hdfs_path", "hdfs_data_path")
        print('start to read actLog_test_single_cross')
        test_file_path = rootPath + 'actLog_test_single_cross'
        actLog_test_rdd = self.sc.pickleFile(test_file_path)
        #labels需要修改
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
        # df_actLog_test.printSchema()


        print('start to read actLog_train_step2')
        train_file_path = rootPath + 'actLog_train_single_cross'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        # print(actLog_train_rdd.take(5))
        df_actLog_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)


        print("对duration_time和time_day 根据finish、like进行分组")
        def DurationLikeBin(x):
            if x <=2:
                return 1
            elif 2<x<=12:
                return 2
            elif 12<x<=15:
                return 3
            elif 15<x<=22:
                return 4
            elif 22<x<=42:
                return 5
            else:
                return 6
        converDurationLikeBin=udf(lambda x :DurationLikeBin(x), typ.IntegerType())
        df_actLog_train = df_actLog_train.withColumn("duration_time_bin_like", converDurationLikeBin(df_actLog_train.duration_time))
        df_actLog_test = df_actLog_test.withColumn("duration_time_bin_like", converDurationLikeBin(df_actLog_test.duration_time))

        def DurationFinishBin(x):
            if x <=2:
                return 1
            elif 2<x<=12:
                return 2
            elif 12<x<=26:
                return 3
            elif 26<x<=42:
                return 4
            else:
                return 5
        converDurationFinishBin=udf(lambda x :DurationFinishBin(x), typ.IntegerType())
        df_actLog_train = df_actLog_train.withColumn("duration_time_bin_finish", converDurationFinishBin(df_actLog_train.duration_time))
        df_actLog_test = df_actLog_test.withColumn("duration_time_bin_finish", converDurationFinishBin(df_actLog_test.duration_time))


        def TimeLikeBin(x):
            if x >=822:
                return 1
            elif 810<=x<822:
                return 2
            elif 781<=x<810:
                return 3
            elif 748<=x<781:
                return 4
            elif 726<=x<748:
                return 5
            elif 646<=x<726:
                return 6
            else:
                return 7

        converTimeLikeBin=udf(lambda x :TimeLikeBin(x), typ.IntegerType())
        df_actLog_train = df_actLog_train.withColumn("time_day_bin_like", converTimeLikeBin(df_actLog_train.time_day))
        df_actLog_test = df_actLog_test.withColumn("time_day_bin_like", converTimeLikeBin(df_actLog_test.time_day))


        def TimeFinshBin(x):
            if x >=795:
                return 1
            elif 792<=x<795:
                return 2
            elif 632<=x<792:
                return 3
            else:
                return 4

        converTimeFinshBinBin=udf(lambda x :TimeFinshBin(x), typ.IntegerType())
        df_actLog_train = df_actLog_train.withColumn("time_day_bin_finish", converTimeFinshBinBin(df_actLog_train.time_day))
        df_actLog_test = df_actLog_test.withColumn("time_day_bin_finish", converTimeFinshBinBin(df_actLog_test.time_day))

        #删除原始列
        print("删除没有必要的列")
        unuse_col=['item_city','user_city','device','author_id','music_id',"duration_time","time_day"]  #'uid','item_id'这两列不能删除，后面提交结果的时候应该要用到
        unuse_col=unuse_col+['item_pub_month','item_pub_day','item_pub_minute']
        df_actLog_train=self.dropUnuseCols(df_actLog_train,unuse_col)
        df_actLog_test=self.dropUnuseCols(df_actLog_test,unuse_col)

        # df_actLog_train=df_actLog_train.drop("duration_time").drop("time_day")
        # df_actLog_test=df_actLog_test.drop("duration_time").drop("time_day")
        # df_actLog_train.show(1,truncate=False)
        # df_actLog_train.printSchema()

        print('start to read nlp_topic_feature2')
        nlp_file_path = rootPath + 'nlp_topic_feature2'
        nlp_topic_rdd = self.sc.pickleFile(nlp_file_path)
        # item_id|title_features |title_words_unique|title_length|title_features_1 |title_topic|
        df_nlp_topic=nlp_topic_rdd.toDF(['item_id',"title_features",'title_words_unique','title_length',"title_features_1",'title_topic'])
        #删除无用列
        df_nlp_topic=df_nlp_topic.drop("title_features")
        df_nlp_topic=df_nlp_topic.drop("title_features_1")
        # df_nlp_topic.show(2)
        # df_nlp_topic.printSchema()


        print("start to read face_feature")
        face_file_path = rootPath + 'face_feature'
        face_rdd = self.sc.pickleFile(face_file_path)
        labels=[
                ('item_id',typ.IntegerType()),
                ('gender',typ.IntegerType()),
                ('beauty',typ.DoubleType()),
                ('relative_position_0',typ.DoubleType()),
                ('relative_position_1',typ.DoubleType()),
                ('relative_position_2',typ.DoubleType()),
                ('relative_position_3',typ.DoubleType())
            ]
        faceSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df_face = sqlContext.createDataFrame(face_rdd,faceSchema)
        #对所有这些列控制小数位数
        df_face=df_face.withColumn('relative_position_0',fn.bround('relative_position_0', scale=3))
        df_face=df_face.withColumn('relative_position_1',fn.bround('relative_position_1', scale=3))
        df_face=df_face.withColumn('relative_position_2',fn.bround('relative_position_2', scale=3))
        df_face=df_face.withColumn('relative_position_3',fn.bround('relative_position_3', scale=3))

        # df_face=df_face.repartition(300)

        print('start to read uid_item_face_feature')
        face_trainfile_path = rootPath + 'df_uid_face_train'
        face_testfile_path = rootPath + 'df_uid_face_test'

        face_trainrdd = self.sc.pickleFile(face_trainfile_path)
        face_testrdd = self.sc.pickleFile(face_testfile_path)
        #在concate的时候关于item_id和uid进行关联
        #labels也需要修改一下
        labels=[('uid',typ.IntegerType()),
                ('uid_max_beauty',typ.DoubleType()),
                ('uid_avg_beauty',typ.DoubleType()),
                ('uid_male_ratio',typ.DoubleType())
            ]
        itemfaceSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df_face_train = sqlContext.createDataFrame(face_trainrdd,itemfaceSchema)
        df_face_test = sqlContext.createDataFrame(face_testrdd,itemfaceSchema)
        # 去重前记录条数 2761799
        # 去重后记录条数 32615
        df_face_train=df_face_train.dropDuplicates()
        df_face_test=df_face_test.dropDuplicates()

        df_face_train=df_face_train.withColumn('uid_max_beauty',fn.bround('uid_max_beauty', scale=3))
        df_face_train=df_face_train.withColumn('uid_avg_beauty',fn.bround('uid_avg_beauty', scale=3))
        df_face_train=df_face_train.withColumn('uid_male_ratio',fn.bround('uid_male_ratio', scale=3))
        # df_face.show()
        # df_face_train.printSchema()

        print("三表进行关联")   #三个表的数据量不大，但是关联后数据量却比df_actLog_test增加近1000倍
        df_test=df_actLog_test.join(df_nlp_topic,'item_id','left')\
                      .join(df_face,'item_id','left')\
                      .join(df_face_test,"uid",'left')

        df_train=df_actLog_train.join(df_nlp_topic,'item_id','left')\
                       .join(df_face,'item_id','left')\
                      .join(df_face_train,"uid",'left')

        # df_train.show(1,truncate=False)
        print("查看表结构")
        print("schema,为下一步build_data读取数据做准备")
        df_train.printSchema()
        df_test.printSchema()

        print("--------观察新增列uid_max_beauty,uid_avg_beauty,uid_male_ratio是否存在缺失值----")

        # print('查看训练集中每一列的缺失比例')
        # music_id_count_bin_missing|music_id_count_ratio_missing  0.64   因为含有-1值
        #
        # df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()
        #
        # print('查看测试集中每一列的缺失比例')
        #
        # df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()

        #三表关联后，有些item_id是没有title的，导致title部分数据可能存在nan值，这里要进行缺失值填充
        #类别变量填充-1，连续变量用均值填充
        #对'title_topic'，'title_words_unique','title_length'这一列填充-1即可

        #这里music_id_count_bin应该在之前已经填充好了，可以去掉
        df_train=df_train.na.fill({'music_id_count_bin':-1,'music_id_count_ratio':0,\
                                   'title_words_unique':-1,'title_length':-1,'title_topic': -1})
        df_test=df_test.na.fill({'music_id_count_bin':-1,'music_id_count_ratio':0,\
                                 'title_words_unique':-1,'title_length':-1,'title_topic': -1})

        #对face特征进行处理
        #关于uid进行分组，即用户看的每个用户看的所有item_id中max_beauty,avg_beauty
        #对连续变量填充缺失值
        print('输出各均值')
        mean_beauty=0.53
        mean_relative_position0=0.392
        mean_relative_position1=0.228
        mean_relative_position2=0.212
        mean_relative_position3=0.164
        mean_max_beauty=0.792
        mean_avg_beauty=0.53
        #下面注释部分是均值的计算，运行中为减少内存消耗和加快训练速度，直接把之前计算的结果用上了

        # df=df_train.union(df_test)
        # mean_val = df.select(fn.mean(df['beauty'])).collect()
        # mean_beauty = round(mean_val[0][0],3) # to show the number
        # print(mean_beauty)
        # mean_val = df.select(fn.mean(df['relative_position_0'])).collect()
        # mean_relative_position0 = round(mean_val[0][0],3) # to show the number
        # print(mean_relative_position0)
        # mean_val = df.select(fn.mean(df['relative_position_1'])).collect()
        # mean_relative_position1 = round(mean_val[0][0] ,3)# to show the number
        # print(mean_relative_position1)
        # mean_val = df.select(fn.mean(df['relative_position_2'])).collect()
        # mean_relative_position2 = round(mean_val[0][0],3) # to show the number
        # print(mean_relative_position2)
        # mean_val = df.select(fn.mean(df['relative_position_3'])).collect()
        # mean_relative_position3 = round(mean_val[0][0],3) # to show the number
        # print(mean_relative_position3)
        # mean_val = df.select(fn.mean(df['uid_max_beauty'])).collect()
        # mean_max_beauty = round(mean_val[0][0],3) # to show the number
        # print(mean_max_beauty)
        # mean_val = df.select(fn.mean(df['uid_avg_beauty'])).collect()
        # mean_avg_beauty = round(mean_val[0][0],3) # to show the number
        # print(mean_avg_beauty)


        # del df
        gc.collect()

        df_train=df_train.na.fill({'gender': -1, 'beauty': mean_beauty,'relative_position_0': mean_relative_position0, \
                       'relative_position_1': mean_relative_position1,'relative_position_2': mean_relative_position2,\
                       'relative_position_3': mean_relative_position3 ,
                        'uid_max_beauty':mean_max_beauty, 'uid_avg_beauty':mean_avg_beauty, 'uid_male_ratio':0.5})
        df_test=df_test.na.fill({'gender': -1, 'beauty': mean_beauty,'relative_position_0': mean_relative_position0, \
                       'relative_position_1': mean_relative_position1,'relative_position_2': mean_relative_position2,\
                       'relative_position_3': mean_relative_position3 ,
                        'uid_max_beauty':mean_max_beauty, 'uid_avg_beauty':mean_avg_beauty, 'uid_male_ratio':0.5})

        #
        print('填充缺失以后')
        # print('查看训练集中每一列的缺失比例')
        # df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()
        # print('查看测试集中每一列的缺失比例')
        # df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()
        '''
        print("三表关联后的数据保存在hdfs")
        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'df_concate_test'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_test.rdd.map(tuple).saveAsPickleFile(file_path)
        print('文件大小如下')
        os.system("hadoop fs -du -s -h  {}".format(file_path))

        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'df_concate_train'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train.rdd.map(tuple).saveAsPickleFile(file_path)
        print("hdfs保存结束")
        print('文件大小如下')
        os.system("hadoop fs -du -s -h  {}".format(file_path))
        '''




        #以下代码会报错：java.net.SocketException: Connection reset
        #报错
        # Total size of serialized results of 162 tasks (1029.9 MB) is bigger than spark.driver.maxResultSize (1024.0 MB)
        print("三表关联后的数据保存到本地")
        localPath='/data/code/DeepCTR/data/dataForSkearn/'
        df_test.toPandas().to_csv(localPath+"test.csv",index=False)
        df_train.toPandas().to_csv(localPath+"train.csv",index=False)
        print("本地保存结束")

        #如果报错就分批保存  数据量太大了

        #return df_train,df_test



if __name__ == "__main__":
    spark_job = SparkFEProcess()
    # df_train,df_test=
    spark_job.data_describe()