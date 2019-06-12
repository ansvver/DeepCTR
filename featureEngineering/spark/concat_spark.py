#!/usr/local/bin/python

from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import  configparser
import pyspark.sql.types as typ
import os
import gc
import pyspark.sql.functions as fn
# import pyspark.sql.functions  as psf
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import StandardScaler, VectorAssembler
import pickle

class SparkFEProcess:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark") \
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

    def data_describe(self):
        sqlContext = SQLContext(self.sc)
        rootPath=self.parser.get("hdfs_path", "hdfs_data_path")
        print('start to read actLog_test_bin')
        test_file_path = rootPath + 'actLog_test_bin'
        actLog_test_rdd = self.sc.pickleFile(test_file_path)
        # print(actLog_test_rdd.take(5))
        #2108547, 3909, 0, -1, -1, 9, '1', 6, 0, 4, 5, 0, 3, 2)
        labels=[
                ('item_id',typ.IntegerType()),
                ('uid',typ.IntegerType()),
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
                ('item_city_score_bin',typ.IntegerType())
            ]
        actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])

        df_actLog_test = sqlContext.createDataFrame(actLog_test_rdd,actionLogSchema)
        # df_actLog_test.show(truncate=False)
        # df_actLog_test.printSchema()

        print('start to read actLog_train_bin')
        train_file_path = rootPath + 'actLog_train_bin'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        # print(actLog_train_rdd.take(5))
        df_actLog_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)
        # df_actLog_train.show(truncate=False)
        # df_actLog_train.printSchema()

        print('start to read nlp_topic_feature2')
        nlp_file_path = rootPath + 'nlp_topic_feature2'
        nlp_topic_rdd = self.sc.pickleFile(nlp_file_path)
        #'item_id', 'title_topic'
        # df_nlp_topic=nlp_topic_rdd.toDF(['item_id','topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9',\
        #                                  'topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19',\
        #                                  'topic20','topic21','topic22','topic23','topic24','topic25','topic26','topic27','topic28','topic29',\
        #                                  'topic30','topic31','topic32','topic33','topic34','topic35','topic36','topic37','topic38','topic39',\
        #                                  'topic40','topic41','topic42','topic43','topic44','topic45','topic46','topic47','topic48','topic49'])
        df_nlp_topic=nlp_topic_rdd.toDF(['item_id','title_topic'])
        df_nlp_topic.show(2)
        # df_nlp_topic.printSchema()


        print('start to read face_feature')
        face_file_path = rootPath + 'face_feature'
        face_rdd = self.sc.pickleFile(face_file_path)
        # print(face_rdd.take(5))
        # item_id': 813408, 'gender': None, 'beauty','relative_position_0', 0), ('relative_position_1', 1), ('relative_position_2', 2), ('relative_position_3', 3
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
        # df_face.show()
        # df_face.printSchema()

        #三表进行关联
        df_test=df_actLog_test.join(df_nlp_topic,'item_id','left')\
                      .join(df_face,'item_id','left')
        df_train=df_actLog_train.join(df_nlp_topic,'item_id','left')\
                      .join(df_face,'item_id','left')


        print('查看训练集中每一列的缺失比例')
        # df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()
        print('查看测试集中每一列的缺失比例')
        # df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()


        #三表关联后，有些item_id是没有title的，导致title部分数据可能存在nan值，这里要进行缺失值填充

        #类别变量填充-1，连续变量用均值填充
        #对'title_topic'这一列填充-1即可
        df_train=df_train.na.fill({'title_topic': -1})
        df_test=df_test.na.fill({'title_topic': -1})


        #对连续变量填充缺失值
        print('输出各均值')
        df=df_train.union(df_test)
        mean_val = df.select(fn.mean(df['beauty'])).collect()
        mean_beauty = mean_val[0][0] # to show the number
        print(mean_beauty)
        mean_val = df.select(fn.mean(df['relative_position_0'])).collect()
        mean_relative_position0 = mean_val[0][0] # to show the number
        print(mean_relative_position0)
        mean_val = df.select(fn.mean(df['relative_position_1'])).collect()
        mean_relative_position1 = mean_val[0][0] # to show the number
        print(mean_relative_position1)
        mean_val = df.select(fn.mean(df['relative_position_2'])).collect()
        mean_relative_position2 = mean_val[0][0] # to show the number
        print(mean_relative_position2)
        mean_val = df.select(fn.mean(df['relative_position_3'])).collect()
        mean_relative_position3 = mean_val[0][0] # to show the number
        print(mean_relative_position3)

        del df
        gc.collect()

        df_train=df_train.na.fill({'gender': -1, 'beauty': mean_beauty,'relative_position_0': mean_relative_position0, \
                       'relative_position_1': mean_relative_position1,'relative_position_2': mean_relative_position2,\
                       'relative_position_3': mean_relative_position3})
        df_test=df_test.na.fill({'gender': -1, 'beauty': mean_beauty,'relative_position_0': mean_relative_position0, \
                       'relative_position_1': mean_relative_position1,'relative_position_2': mean_relative_position2,\
                       'relative_position_3': mean_relative_position3})

        print('填充缺失以后')
        print('查看训练集中每一列的缺失比例')
        # df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()
        print('查看测试集中每一列的缺失比例')
        # df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()
        print("schema")
        df_train.printSchema()
        df_test.printSchema()

        print("三表关联后的数据保存在hdfs")
        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'df_train'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_train.rdd.map(tuple).saveAsPickleFile(file_path)

        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'df_test'
        os.system("hadoop fs -rm -r {}".format(file_path))  #os.system(command) 其参数含义如下所示: command 要执行的命令
        df_test.rdd.map(tuple).saveAsPickleFile(file_path)
        print("hdfs保存结束")


        #以下代码会报错：java.net.SocketException: Connection reset
        print("三表关联后的数据保存到本地")
        localPath='/data/code/DeepCTR/data/dataForSkearn/'
        df_train.toPandas().to_csv(localPath+"train.csv",index=False)
        df_test.toPandas().to_csv(localPath+"test.csv",index=False)
        print("本地保存结束")





        #return df_train,df_test



if __name__ == "__main__":
    spark_job = SparkFEProcess()
    # df_train,df_test=
    spark_job.data_describe()

