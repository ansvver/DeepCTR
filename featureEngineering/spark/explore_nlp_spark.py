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


        print('dataframe finish, start to deal with the title_features col')

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
        # df.printSchema()
        # df.show(10)
        #
        # # 执行下面这步前需要将title_features转化为向量，同HashingTF后输出的向量格式相同
        # idf = IDF(inputCol="title_features_2", outputCol="features")
        # idfModel = idf.fit(df)
        # df = idfModel.transform(df)
        # df.show()
        # df.printSchema()


        #title_features_idf=SparseVector(134545, {1: 3.2217, 2: 8.5988, 3: 2.0428, 4: 1.3045, 5: 3.8285, 6: 6.2922})
        # rescaledData=df.select("item_id", "features")
        del df
        gc.collect()
        # rescaledData.show(truncate=False)
        # rescaledData = rescaledData.filter(rescaledData['item_id']<338441)
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

        # print('查看列名')
        # print([psf.col("topic")[i] for i in range(50)])
        # print(psf.col("topic")[0])
        df_topic=transformed.withColumn("topic", to_array(psf.col("topicDistribution"))).select(["item_id"] + [psf.col("topic")[i] for i in range(50)])
        # df_topic.show(1,truncate=False)
        # print(df_topic.columns)

        # DataFrame转换成RDD
        # result = df_topic.rdd.map(lambda p: "topic[0]: " + p[psf.col("topic")[0]] + "topic[0]: " + p[psf.col("topic")[1]]).take(2)

        #打印RDD数据
        # for n in result:
        #     print(n)

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

            # valueList=[]
            # for c in topicCol: #第一次遍历找出topic列中的最大值
            #     valueList.append(p[c])
            # maxVal=max(valueList)
            # for c in topicCol:#第二次遍历找到哪个列为最大值
            #     if maxVal==p[c]:
            #        return int(c.replace("topic[",'').replace("]",''))

        # print(df_topic.rdd.map(lambda p: (p.item_id,p['topic[1]'])).take(2))
        # [(4036886, 0.08152284314414694), (2893187, 0.7572771693261391)]

        df_topic1=df_topic.rdd.map(lambda p: (p.item_id, getTopicID(p)))

        # print('观看df_topic1，是不是有些元素不能返回topicId')
        # print(df_topic1.take(5))
        labels=[
            ('item_id',typ.IntegerType()),
            ('title_topic',typ.IntegerType())]
        Schema=typ.StructType([typ.StructField(e[0],e[1],True) for e in labels])
        df_topic2 = sqlContext.createDataFrame(df_topic1,Schema)

        # df_topic2 = df_topic1.toDF(['item_id','topic'])
        # print('观看topic是否为想要的数据格式，并保存于topic2中')
        df_topic2.show(5)



        # topicCol=['topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9',\
        #                                  'topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19',\
        #                                  'topic20','topic21','topic22','topic23','topic24','topic25','topic26','topic27','topic28','topic29',\
        #                                  'topic30','topic31','topic32','topic33','topic34','topic35','topic36','topic37','topic38','topic39',\
        #                                  'topic40','topic41','topic42','topic43','topic44','topic45','topic46','topic47','topic48','topic49',]
        #
        # df_topic_max=df_topic_max_rdd.toDF(['item_id']+topicCol
        #                                  ['topicMax'])
        # def getColName(p):
        #     for c in topicCol:
        #         if p['topicMax']==p[c]:
        #             return c
        #
        #
        #
        # df_topic_max.rdd.map(lambda p: (p.item_id,getColName(p))).toDF(['item_id','topic'])

        #于是topic是一个类别变量，有50个类别

        #保存df_topic
        print('-------5.保存数据预处理结果-------')
        file_path = self.parser.get("hdfs_path", "hdfs_data_path") + 'nlp_topic_feature2'
        os.system("hadoop fs -rm -r {}".format(file_path))
        df_topic2.rdd.map(tuple).saveAsPickleFile(file_path)
        print('数据保存结束')


        #检验上面创建lda模型中使用的参数 ll越大越好，lp越小越好
        '''
        ll = ldaModel.logLikelihood(rescaledData)
        lp = ldaModel.logPerplexity(rescaledData)
        print(ll)
        print(lp)
        '''

        #保存ldaModel,训练集转化的时候直接加载该模型,目前没有必要保存模型，保存df_topic即可
        '''
        distributed_model_path = self.parser.get("hdfs_path", "hdfs_data_path") + "lda_distributed_model"
        ldaModel.save(distributed_model_path)
        #加载的语句
        sameLdaModel = DistributedLDAModel.load(distributed_model_path)
        '''
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