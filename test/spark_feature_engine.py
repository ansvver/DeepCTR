import os
import  sys

current_path = os.path.dirname(os.path.realpath(__file__))
workspace_path = current_path.split('featureEngineering')[0]
sys.path.append(os.path.join(workspace_path, 'test'))

from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
from pyspark.sql.types import *
import  configparser
import  pyspark.sql.types as types
from pyspark.sql import  functions
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import StandardScaler
import  json

sys.path.append("..")

'''
spark计算引擎，负责数据读取 存储 转换等底层基础任务，支撑业务层的特征工程计算需求
'''

class SparkFeatureEngine:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark") \
            .set("spark.ui.showConsoleProgress", "false")
        self.sc = SparkContext(conf=sparkConf)
        self.sc.broadcast(self.parser)
        self.sqlContext = SQLContext(self.sc)
        # self.init_logger()

    def init_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        workspace_path = current_path.split('featureEngineering')[0]
        config_file = workspace_path + 'resource/config.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file)
        return  parser

    def get_broadcastValue(self, value):
        return self.sc.broadcast(value)

    def read_rdd(self, fileName):
        try:
            #file_path = self.parser.get("hdfs_path", "hdfs_data_path") + fileName
            file_path = fileName
            data_rdd = self.sc.textFile(file_path)
            return data_rdd
        except Exception as e:
            print(e)


    def rdd2df(self, rdd, schema):
        try:
            df = self.sqlContext.createDataFrame(rdd, schema)
            return df
        except Exception as e:
            print(e)




if __name__ == "__main__":
    train_data_sample = sys.argv[1]

    spark_job = SparkFeatureEngine()

    # spark_job.build_train_input(train_data_sample)
    #
    # spark_job.sc.stop()


