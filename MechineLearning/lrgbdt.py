from numpy import allclose
import  configparser
import os
import sys
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from  pyspark.sql import  *
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
import datetime
from pyspark.sql.functions import udf
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor,GBTRegressionModel
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
# from pyspark.mllib.tree.model import Node
from pyspark.ml.classification import GBTClassifier,GBTClassificationModel,LogisticRegression

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import numpy as np
import gc



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
        workspace_path = current_path.split('machineLearning')[0]
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

        print('start to read nlp_topic_feature')
        nlp_file_path = rootPath + 'nlp_topic_feature'
        nlp_topic_rdd = self.sc.pickleFile(nlp_file_path)
        # print(nlp_topic_rdd.take(5))
        # 'item_id', 'topic[0]', 'topic[1]', 'topic[2]', 'topic[3]', 'topic[4]......topic[49]
        #
        df_nlp_topic=nlp_topic_rdd.toDF(['item_id','topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9',\
                                         'topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19',\
                                         'topic20','topic21','topic22','topic23','topic24','topic25','topic26','topic27','topic28','topic29',\
                                         'topic30','topic31','topic32','topic33','topic34','topic35','topic36','topic37','topic38','topic39',\
                                         'topic40','topic41','topic42','topic43','topic44','topic45','topic46','topic47','topic48','topic49'])

        # df_nlp_topic.show()
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
        return df_train,df_test



    def data_explore(self,train,test):
        sqlContext = SQLContext(self.sc)
        #处理数据
        cols=train.columns
        cols.remove('like')
        cols.remove('finish')
        target1=['like']
        taeget2=['finish']
        feature1=target1+cols
        train=train.select(feature1)
        test=test.select(feature1)
        # train.printSchema()
        # train.describe(['like']).show()


        #将数据类型转化为double型
        # for i in train.columns:
        #     df=df.withColumn(i, df[i].cast('double'))

        # print(df.printSchema())

        #转化为模型支持的数据格式
        row = Row('label','features')

        trainingData = train.rdd.map(lambda r: (row(r[0],Vectors.dense(r[1:])))).toDF()
        #将trainSet一分为二，一半用于训练GBDT模型，一般用于预测
        # trainSet1=trainSet.limit(9811170)  #总数为19622340
        # trainSet2=trainSet.subtract(trainSet1)

        testData = test.rdd.map(lambda r: (row(r[0],Vectors.dense(r[1:])))).toDF()

        # data=trainingData.union(testData)

        # labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(trainingData)
        stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = stringIndexer.fit(trainingData)
        td = si_model.transform(trainingData)
        # Automatically identify categorical features, and index them.
        # Set maxCategories so features with > 4 distinct values are treated as continuous.
        # featureIndexer =\
        #     VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=10).fit(trainingData)

        # Split the data into training and test sets (30% held out for testing)
        (trainingData1, trainingData2) = trainingData.randomSplit([0.7, 0.3])

        # Train a GBT model.
        gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=5, maxDeth=2,cacheNodeIds=True)

        # Chain indexers and GBT in a Pipeline
        # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

        # Train model.  This also runs the indexers.
        model = gbt.fit(td)
        # Make predictions.
        predictions = model.transform(td)

        # |label|features|indexed|rawPrediction|probability|prediction|
        predictions.select("indexed", "rawPrediction", "probability").show(1,truncate=False)

        # Select example rows to display.
        # predictions.select("prediction", "label", "features").show(1,truncate=False)
        # print(model.treeWeights)  #[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # print(model.getNumTrees)  #10 每一次迭代将产生一棵,增加maxiter会减少模型偏差，但相应的会增加过拟合概率，
        # print(model.totalNumNodes)
        # print(model.toDebugString)
        #获取叶子节点
        print(gbt.getCacheNodeIds())  #True


        totalColNum=model.totalNumNodes  #630,这是gbdt转化后的全部特征个数630个，onehot编码，有10个元素为1，其余620个元素全为0
        GBTMaxIter = len(model.trees)
        print(GBTMaxIter)   #10棵树
        for tree in model.trees:
            print(tree.depth)
            print(tree.numNodes) #每棵树有63个节点
            leafNum=np.math.pow(2,tree.depth)
            print(leafNum)

        # print(model.featureImportances)
        # print(model.trees[0].)







        # print('获取叶子节点编号')
        # print(model.getCacheNodeIds())

        # Select (prediction, true label) and compute test error
        # evaluator = MulticlassClassificationEvaluator(
        #     labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
        # accuracy = evaluator.evaluate(predictions)
        # print("Test Error = %g" % (1.0 - accuracy))
        #
        # gbtModel = model.stages[2]
        # print(gbtModel)  # summary only


        # gbtc_path = temp_path + "gbtc"
        # gbt.save(gbtc_path)
        # gbt2 = GBTClassifier.load(gbtc_path)
        # gbt2.getMaxDepth()
        #
        # model_path = temp_path + "gbtc_model"
        # model.save(model_path)
        # model2 = GBTClassificationModel.load(model_path)
        # model.featureImportances == model2.featureImportances
        #
        # model.treeWeights == model2.treeWeights
        #
        # model.trees


if __name__ == "__main__":
    spark_job = SparkFEProcess()

    df_train,df_test=spark_job.data_describe()

    spark_job.data_explore(df_train,df_test)