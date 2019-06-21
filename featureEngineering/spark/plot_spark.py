___author__ = 'zlx'

import  configparser
import os
from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import pyspark.sql.types as typ
import matplotlib.pyplot as plt

class SparkFEProcess:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark of plot") \
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

    def data_plot(self):
        print('start to read data for rdd:')
        sqlContext = SQLContext(self.sc)
        rawRdd_train = self.read_rdd('final_track2_train.txt').map(lambda line : line.split('\t'))
        # rawRdd_test = self.read_rdd('final_track2_test_no_anwser.txt').map(lambda line : line.split('\t'))
        print('finish read rdd, start to init action log rdd:')
        duration_times=rawRdd_train.map(
        lambda x :( int(x[0]),int(x[7]), int(x[11])))

        labels=[('uid',typ.IntegerType()),
            ('like',typ.IntegerType()),
            ('duration_time',typ.IntegerType())]

        Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df = sqlContext.createDataFrame(duration_times, Schema)

        pd=df.toPandas()
        # print(pd)

        groups=pd.groupby(['duration_time','like']).count()
        # print(groups)
        tmp = groups['uid']
        # print("选一列")
        # print(tmp)

        tmp2 = tmp.unstack()
        # print("反向")
        # print(tmp2 )

        tmp2.plot(kind='bar')
        plt.show()




if __name__ == "__main__":
    spark_job = SparkFEProcess()

    spark_job.data_plot()


