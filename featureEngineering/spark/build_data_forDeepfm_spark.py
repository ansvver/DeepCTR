#!/usr/local/bin/python

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

        sparkConf = SparkConf().setAppName("build data for deepfm") \
            .set("spark.ui.showConsoleProgress", "false") \
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

    def build_data(self):
        sqlContext = SQLContext(self.sc)
        rootPath=self.parser.get("hdfs_path", "hdfs_data_path")
        print('start to read concate data of test')
        test_file_path = rootPath + 'df_concate_test'
        actLog_test_rdd = self.sc.pickleFile(test_file_path)
        #修改labels，根据前一个文件的printSchema信息
        labels=[('uid',typ.IntegerType()),
                ('item_id',typ.IntegerType()),
                ("author_id_item_city_music_id_item_pub_hour",typ.StringType()),
                ("uid_user_city_channel_device",typ.StringType()),
                ("author_id_item_pub_hour",typ.StringType()),
                ("author_id_music_id",typ.StringType()),
                ("author_id_item_city",typ.StringType()),
                ("author_id_user_city",typ.StringType()),
                ("author_id_channel",typ.StringType()),
                ("uid_item_pub_hour",typ.StringType()),
                ("uid_device",typ.StringType()),
                ("uid_music_id",typ.StringType()),
                ("uid_channel",typ.StringType()),
                ("uid_item_city",typ.StringType()),
                ("uid_author_id",typ.StringType()),
                ("uid_user_city",typ.StringType()),
                ("uid_item_id",typ.StringType()),

                ('channel',typ.IntegerType()),
                ('finish',typ.IntegerType()),
                ('like',typ.IntegerType()),
                ('duration_time',typ.IntegerType()),
                ('item_pub_hour',typ.IntegerType()),
                ('device_Cnt_bin',typ.IntegerType()),
                ('authorid_Cnt_bin',typ.IntegerType()),
                ('musicid_Cnt_bin',typ.IntegerType()),
                ('uid_playCnt_bin',typ.IntegerType()),
                ('itemid_playCnt_bin',typ.IntegerType()),
                ('user_city_score_bin',typ.IntegerType()),
                ('item_city_score_bin',typ.IntegerType()),
                ('title_topic',typ.IntegerType()),
                ('gender',typ.IntegerType()),
                ('beauty',typ.DoubleType()),
                ('relative_position_0',typ.DoubleType()),
                ('relative_position_1',typ.DoubleType()),
                ('relative_position_2',typ.DoubleType()),
                ('relative_position_3',typ.DoubleType())
            ]
        actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df_test = sqlContext.createDataFrame(actLog_test_rdd,actionLogSchema)

        print('start to read concate data of train')
        train_file_path = rootPath + 'df_concate_train'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        df_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)

        df_train_count=df_train.count()
        print(df_train_count)
        print('df_train_count:19622340')     #全部记录数：22384139
        print('df_test_count:2761799')

        localPath='/data/code/DeepCTR/data/dataForDeepfmTest618/'
        train_label = df_train['finish','like']
        test_label = df_test['finish','like']
        print('保存test_label')
        test_label.toPandas().to_csv(localPath+"test_label.csv",index=False)
        print('保存test_label结束')

        df=df_train.union(df_test)
        # df.cache()  #不能用cache(),本身内存不够，再将df保存在内存中，会导致内存溢出
        feat_dict = {}
        cnt = 1
        #需要处理的特征
        #类别型特征如下：
        ca_col = [ 'channel','time_day','item_pub_month','item_pub_day','item_pub_hour','item_pub_minute',\
                'uid_count_bin','user_city_count_bin','item_id_count_bin','author_id_count_bin','item_city_count_bin',\
                'music_id_count_bin',  'device_count_bin',  'duration_time_count_bin',   'uid_item_id_count_bin',\
                'uid_author_id_count_bin','uid_item_city_count_bin',  'uid_channel_count_bin',     'uid_music_id_count_bin',\
                'uid_device_count_bin',   'uid_item_pub_hour_count',  'author_id_channel_count_bin', 'author_id_user_city_count_bin',\
                'author_id_item_city_count_bin','author_id_music_id_count_bin','author_id_item_pub_hour_count_bin',\
                'uid_user_city_channel_device_count_bin','author_id_item_city_music_id_item_pub_hour_count_bin',\

                'title_words_unique','title_length', 'title_topic',\
                'gender']
        #连续型特征如下：
        co_col=['uid_count_ratio','user_city_count_ratio','item_id_count_ratio','author_id_count_ratio','item_city_count_ratio',\
                'music_id_count_ratio','device_count_ratio','duration_time_count_ratio','uid_item_id_count_ratio',\
                'uid_author_id_count_ratio','uid_item_city_count_ratio','uid_channel_count_ratio','uid_music_id_count_ratio',\
                'uid_device_count_ratio','uid_item_pub_hour_count_ratio','author_id_channel_count_ratio','author_id_user_city_count_ratio',\
                'author_id_item_city_count_ratio','author_id_music_id_count_ratio','author_id_item_pub_hour_count_ratio',\
                'uid_user_city_channel_device_count_ratio','author_id_item_city_music_id_item_pub_hour_count_ratio',\

                'beauty', 'relative_position_0','relative_position_1','relative_position_2','relative_position_3',\
                'uid_max_beauty','uid_avg_beauty','uid_male_ratio']

                #这部分是连续变量，缺失部分已经用均值填充过了，同时其最大最小值都在0-1之间，因此不需要再做额外的处理

        #不参与训练的列 ：item_id\uid
        #对连续变量不做处理，因为duration_time的值范围已经是[0,300]了

        #无需处理的连续变量
        for col in co_col:
            feat_dict[col] = cnt  #index
            cnt += 1

        #对类别变量进行处理
        for col in ca_col:
            us=df.select(col).distinct().rdd.map(lambda r: r[0]).collect()
            # print(col)
            # print(len(us))
            feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))   #类别变量col里面每个value都分配一个index
            cnt += len(us)

        feat_dim_file = open(localPath+"feat_dim.txt",mode='w',encoding='utf-8')
        feat_dim_file.write(str(cnt))
        feat_dim_file.close()


        ##从这一步开始对df_train 和df_test分别处理
        feature_value_train=df_train.select(co_col+ca_col)
        feature_index_train = feature_value_train
        for i in feature_index_train.columns:
            # print(i)
            # print(feat_dict[i])  #这是一个dict
            if i in co_col:
                #替换已经存在的同名列,返回一个新的dataframe,
                a=feat_dict[i]
                # AssertionError: col should be Column,第二个参数必须是含有列的表达式
                feature_index_train=feature_index_train.withColumn(i, fn.lit(a))
            if i in ca_col:
                k=list(feat_dict[i].keys())
                v=list(feat_dict[i].values())
                feature_index_train=feature_index_train.replace(k,v,i)  #k,v都是list，对i列中的k用v来替代
                feature_value_train=feature_value_train.withColumn(i, fn.lit(1))


        feature_index_train=feature_index_train.coalesce(1).withColumn("id", monotonically_increasing_id())
        feature_value_train=feature_value_train.coalesce(1).withColumn("id", monotonically_increasing_id())
        train_label = train_label.coalesce(1).withColumn("id", monotonically_increasing_id())

        print('-------.开始保存训练数据-------')
        #将训练集划分成20份，
        segList= [i for i in range(0,df_train_count+1,df_train_count//20)]
        # print('train存储方案一：toPandas()后保存到本地')
        for i in range(len(segList)):
            if i <=19:
                # print(i)
                # print(segList[i],segList[i+1])
                train_feature_index_i=feature_index_train.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                train_feature_value_i=feature_value_train.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                train_label_i=train_label.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')

                train_feature_index_i.toPandas().to_csv(localPath+"train_feature_index_"+str(i)+".csv",index=False)
                train_feature_value_i.toPandas().to_csv(localPath+"train_feature_value_"+str(i)+".csv",index=False)
                train_label_i.toPandas().to_csv(localPath+"train_label_"+str(i)+".csv",index=False)




        print("开始处理测试数据")
        feature_value_test=df.select(co_col+ca_col)
        feature_index_test = feature_value_test
        for i in feature_index_test.columns:
            # print(i)
            # print(feat_dict[i])  #这是一个dict
            if i in co_col:
                #替换已经存在的同名列,返回一个新的dataframe,
                a=feat_dict[i]
                # AssertionError: col should be Column,第二个参数必须是含有列的表达式
                feature_index_test=feature_index_test.withColumn(i, fn.lit(a))
            if i in ca_col:
                k=list(feat_dict[i].keys())
                v=list(feat_dict[i].values())
                feature_index_test=feature_index_test.replace(k,v,i)  #k,v都是list，对i列中的k用v来替代
                feature_value_test=feature_value_test.withColumn(i, fn.lit(1))

        print('test数据开始存储：toPandas()后保存到本地')  #这一步没有保存，报错java.net.SocketException: Connection reset
        feature_index_test.toPandas().to_csv(localPath+"test_feature_index.csv",index=False)
        feature_value_test.toPandas().to_csv(localPath+"test_feature_value.csv",index=False)


        print("一次性toPandas()到本地")  #一次性保存，应该会有内存溢出的问题
        feature_index_train.toPandas().to_csv(localPath+"train_feature_index.csv",index=False)
        feature_value_train.toPandas().to_csv(localPath+"train_feature_value.csv",index=False)
        train_label.toPandas().to_csv(localPath+"train_label.csv",index=False)



if __name__ == "__main__":
    spark_job = SparkFEProcess()

    # df_train,df_test=spark_job.data_describe()
    # idx = 0
    spark_job.build_data()    #完成读取数据