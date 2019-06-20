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
from pyspark.ml.feature import VectorAssembler
import datetime
from pyspark.sql.functions import udf
import pyspark.sql.functions  as psf
from pyspark.ml.linalg import Vectors, VectorUDT,SparseVector
import numpy as np
import gc
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Normalizer
from pyspark.ml.clustering import LDA, DistributedLDAModel


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
        print('start to read data for rdd:')
        rawRdd_nlp = self.read_rdd('track2_title.txt').map(lambda line : eval(line))
        # print(rawRdd_nlp.take(10))
        #转化为dataframe,在不指定schema的情况下会自动推断
        sqlContext = SQLContext(self.sc)
        labels=[
            ('item_id',typ.IntegerType()),
            ('title_features',typ.MapType(typ.StringType(), typ.IntegerType()))]
        Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df = sqlContext.createDataFrame(rawRdd_nlp,Schema)
        # df.show(10)
        # df.printSchema()

        print("统计title中不同词的个数unique，以及title的长度")
        gdf=df.select("item_id",fn.explode(fn.col("title_features"))).groupBy("item_id")
        df2=gdf.agg(fn.count("key").alias("title_words_unique"))

        df3=gdf.agg(fn.sum("value").alias("title_length"))

        df=df.join(df2,"item_id","left") \
             .join(df3,"item_id","left")
        df=df.drop("title_features")
        df.printSchema()


        print('start to deal with the title_features col,and compute the title topic')

        tokens=df.rdd.map(lambda d:d[1]).map(lambda d:list(d.keys()))  #每个titile对应的tokens

        local_tokens=tokens.flatMap(lambda d :[int(token) for token in d]).distinct()
        print('local_tokens最大的值')
        print(local_tokens.top(1))
        vocab_size=max(local_tokens.top(1))+1

        #将title_feature列转化为向量
        toInt=udf(lambda counts :{int(token) :float(counts[token]) for token in counts}, typ.StringType())
        df = df.withColumn("title_features_1", toInt(df.title_features))


        toVector=udf(lambda vs: Vectors.sparse(vocab_size,vs), VectorUDT())
        rescaledData = df.withColumn("features", toVector(df.title_features_1)).select("item_id", "features")

        df=df.drop("title_features_1")
        # del df
        # gc.collect()
        rescaledData.cache()
        lda = LDA(k=50,maxIter=200)
        # lda = LDA(k=2,maxIter=5)
        ldaModel = lda.fit(rescaledData)

        transformed = ldaModel.transform(rescaledData)   #.select("topicDistribution")
        #结果显示 每个文档各个类别的权重, transformed表各列名
        #主题分布向量转化为类别
        # transformed.show(truncate=False)

        def to_array(col):
            def to_array_(v):
                return v.toArray().tolist()
            return psf.udf(to_array_, typ.ArrayType(typ.DoubleType()))(col)
        df_topic=transformed.withColumn("topic", to_array(psf.col("topicDistribution"))).select(["item_id"] + [psf.col("topic")[i] for i in range(50)])

        topicCol=df_topic.columns
        topicCol.remove("item_id")
        print('查看列名')
        print(topicCol)
        def getTopicID(p):  #改用key-value的形式，再排序，找出最大value对应的key
            d={}
            for c in topicCol: #构建字典
                d[c]=p[c]
            z = list(d.keys())[list(d.values()).index(max(d.values()))]
            return int(z.replace("topic[",'').replace("]",''))

        df_topic1=df_topic.rdd.map(lambda p: (p.item_id, getTopicID(p)))

        labels=[
            ('item_id',typ.IntegerType()),
            ('title_topic',typ.IntegerType())]
        Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df_topic2 = sqlContext.createDataFrame(df_topic1,Schema)

        # df_topic2 = df_topic1.toDF(['item_id','topic'])
        # print('观看topic是否为想要的数据格式，并保存于topic2中')
        df_topic2.show(5)

        df=df.join(df_topic2,"item_id","left")   #UnboundLocalError: local variable 'df' referenced before assignment
        df.printSchema()
        #item_id|title_features |title_words_unique|title_length|title_features1 |title_topic|

        print('-------5.保存数据预处理结果-------')
        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'nlp_topic_feature2'
        os.system("hadoop fs -rm -r {}".format(file_path))
        df.rdd.map(tuple).saveAsPickleFile(file_path)
        print('数据保存结束')

        #检验上面创建lda模型中使用的参数 ll越大越好，lp越小越好
        '''
        ll = ldaModel.logLikelihood(rescaledData)
        lp = ldaModel.logPerplexity(rescaledData)
        print(ll)
        print(lp)
        '''

        #保存ldaModel,训练集转化的时候直接加载该模型,目前没有必要保存模型，保存df_topic即可

        # distributed_model_path = self.parser.get("hdfs_path", "hdfs_data_path") + "lda_distributed_model"
        # ldaModel.save(distributed_model_path)
        # #加载的语句
        # sameLdaModel = DistributedLDAModel.load(distributed_model_path)

        # ---------------------------------3 模型及描述------------------------------
        # 模型通过describeTopics、topicsMatrix来描述
        '''
        topicIndices = ldaModel.describeTopics(maxTermsPerTopic=5)
        topicIndices.show(truncate=False)
        #*主题    主题包含最重要的词语序号                     各词语的权重
        '''


        '''
        #计算L2norm
        normalizer = Normalizer(inputCol="title_features_idf", outputCol="norm")
        data = normalizer.transform(rescaledData)



        dot_udf = udf(lambda x,y: float(x.dot(y)), typ.DoubleType())
        data.alias("i").join(data.alias("j"), psf.col("i.item_id") < psf.col("j.item_id"))\
            .select(
                psf.col("i.item_id").alias("i"),
                psf.col("j.item_id").alias("j"),
                dot_udf("i.norm", "j.norm").alias("dot"))\
            .sort("i", "j")\
            .show()
        '''






if __name__ == "__main__":
    spark_job = SparkFEProcess()

    spark_job.data_describe()



    # spark_job.data_explore(dfactionLog_train,dfactionLog_test)