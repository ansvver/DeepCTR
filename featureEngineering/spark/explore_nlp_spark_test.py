#!/usr/local/bin/python
import  configparser
import os
import sys
from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.ml.feature import VectorAssembler
import datetime
from pyspark.sql.functions import udf
import pyspark.sql.functions  as psf
from pyspark.ml.linalg import Vectors, VectorUDT,SparseVector
import numpy as np
import gc
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Normalizer
from pyspark.ml.clustering import LDA, LocalLDAModel


class SparkFEProcess:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark of nlp") \
            .set("spark.ui.showConsoleProgress", "false")
        self.sc = SparkContext(conf=sparkConf)
        self.sc.broadcast(self.parser)
        self.init_logger()


    def init_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        workspace_path = current_path.split('featureEngineering')[0]
        config_file = workspace_path + 'resource/config.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file)
        return  parser

    def init_logger(self):
        '''
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
        #rawRdd_nlp = self.read_rdd('track2_title.txt').map(lambda line : eval(line))
        #rawRdd_nlp = self.read_rdd('track2_title_500.txt').map(lambda line : eval(line))
        rawRdd_nlp=self.sc.textFile('/user/hadoop/icmechallenge2019/track2/test/track2_title_500.txt').map(lambda line : eval(line))

        # print(rawRdd_nlp.take(10))

        sqlContext = SQLContext(self.sc)
        labels=[
            ('item_id',typ.IntegerType()),
            ('title_features',typ.MapType(typ.StringType(), typ.IntegerType()))]
        Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df = sqlContext.createDataFrame(rawRdd_nlp,Schema)
        df.show(5)
        # df.printSchema()


        print('start to deal with the title_features col,and compute the title topic')

        tokens = df.rdd.map(lambda d:d[1]).map(lambda d:list(d.keys()))

        local_tokens=tokens.flatMap(lambda d :[int(token) for token in d]).distinct()

        print(local_tokens.top(1))
        vocab_size=max(local_tokens.top(1))+1


        toInt = udf(lambda counts :{ int(token):float(counts[token]) for token in counts}, typ.StringType())
        df = df.withColumn("title_features_1", toInt(df.title_features))


        toVector=udf(lambda vs: Vectors.sparse(vocab_size,vs), VectorUDT())
        rescaledData = df.withColumn("features", toVector(df.title_features_1)).select("item_id", "features")

        rescaledData.cache()
        # lda = LDA(k=50,maxIter=200)
        lda = LDA(k=2,maxIter=5)
        ldaModel = lda.fit(rescaledData)


        print("begin save model")
        distributed_model_path = "/user/hadoop/icmechallenge2019/track2/test/" + "lda_distributed_model_pyspark"
        ldaModel.write().overwrite().save(distributed_model_path)
        print("model saved")

        print("load model")
        sameLdaModel = LocalLDAModel.load(distributed_model_path)
        print("model loaded")

        transformed = sameLdaModel.transform(rescaledData)   #.select("topicDistribution")

        transformed.show(truncate=False)






if __name__ == "__main__":
    spark_job = SparkFEProcess()

    spark_job.data_describe()



    # spark_job.data_explore(dfactionLog_train,dfactionLog_test)