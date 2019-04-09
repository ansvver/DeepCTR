#!/usr/local/bin/python

from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
from pyspark.sql.types import *
import  configparser
import pyspark.sql.types as typ
import os
import pyspark.sql.functions as fn

#from  featureEngineering.settings import *

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
        rawRdd = self.read_rdd('final_track2_train.txt').map(lambda line : line.split('\t'))
        print('finish read rdd, start to init action log rdd:')
        actionLogRdd = rawRdd.map(
            lambda x :(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),
                       int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11])))
        # total = actionLogRdd.count()
        # print('total: ' + str(total))

        #转化为dataframe
        sqlContext = SQLContext(self.sc)
        labels=[('uid',typ.StringType()),
            ('user_city',typ.StringType()),
            ('item_id',typ.StringType()),
            ('author_id',typ.StringType()),
            ('item_city',typ.StringType()),
            ('channel',typ.StringType()),
            ('finish',typ.StringType()),
            ('like',typ.StringType()),
            ('music_id',typ.StringType()),
            ('device',typ.StringType()),
            ('time',typ.StringType()),
            ('duration_time',typ.StringType())]
        actionLogSchema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        dfactionLog = sqlContext.createDataFrame(actionLogRdd, actionLogSchema)

        return dfactionLog


    def data_process(self,df):
        data_dict={}
        feat_dict = {}
        cnt = 1
        # cnt=self.sc.accumulator(1)
        self.sc.broadcast(data_dict)  #声明一个广播变量
        self.sc.broadcast(feat_dict)
        self.sc.broadcast(cnt)

        label = df['finish','like'].toPandas()
        label_finish=label['finish'].values
        label_like=label['like'].values
        print(type(label_finish))

        label_finish = label_finish.reshape(len(label_finish), 1)
        label_like = label_like.reshape(len(label_like), 1)

        data_dict['y_finish'] = label_finish
        data_dict['y_like'] = label_like   #对应的是一个dataframe
        #
        #创建一个空的dataframe
        # co_feature = pd.DataFrame()
        # ca_feature = pd.DataFrame()
        #转化为dataframe
        # sqlContext = SQLContext(self.sc)
        # co_feature = sqlContext.createDataFrame()
        # ca_feature = sqlContext.createDataFrame()
        ca_col = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'device']
        co_col = ['time', 'duration_time']
        #对连续变量做标准化处理
        # for i in range(1, df.shape[1]):   #遍历每一列
        #     target = df.iloc[:, i]
        #     col = target.name
        #     l = len(set(target))
        #     if l > 10:   #连续变量
        #         target = (target - target.mean()) / target.std()  #标准化处理
        #         co_feature = pd.concat([co_feature, target], axis=1)
        #         feat_dict[col] = cnt  #index
        #         cnt += 1
        #         co_col.append(col)

        #对类别变量进行处理
        for col in ca_col:
            us=df.select(col).distinct().rdd.map(lambda r: r[0]).collect()
            # us = df[col].unique()    #sparksql中求一列的值
            print(col)
            print(len(us))
            # print(us)
            feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))   #类别变量col里面每个value都分配一个index
            # ca_feature = pd.concat([ca_feature, target], axis=1)   #sparksql中两个dataframe合并
            cnt += len(us)
            # ca_col.append(col)
        print(cnt)

        feature_value=df[ca_col]  #这一步只是为了创建feature_value这个dataframe，形状(,)

        feature_index = feature_value.copy()
        print('数据列有以下这些')
        print(feature_index.columns)

        for i in feature_index.columns:
            print(i)
            if i in co_col:
                feature_index[i] = feat_dict[i]
            else:
                feature_index[i] = feature_index[i].map(feat_dict[i])
                feature_value[i] = 1.
        # feature_index.to_csv("feature_index.csv", index=None)
        # feature_index.to_csv("feature_value.csv", index=None)
        print(feature_index.show())      #返回的依然是一个dataframe
                                         #取回dataframe中的值
        #以上将训练数据和测试数据一同处理后，再划分

        # train_data_dict=
        pandas_feature_index = feature_index.toPandas()
        pandas_feature_value = feature_value.toPandas()
        data_dict['xi'] = pandas_feature_index.values.tolist()  #变成了list
        data_dict['xv'] = pandas_feature_value.values.tolist()
        data_dict['feat_dim'] = cnt.value
        #将数据预处理后的数据保存
        #保存data_dict ,分别保存训练数据和测试数据



        return data_dict









if __name__ == "__main__":
    spark_job = SparkFEProcess()

    dfactionLog=spark_job.data_describe()

    data_dict=spark_job.data_process(dfactionLog)