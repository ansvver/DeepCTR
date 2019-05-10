#!/usr/local/bin/python

from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
import  configparser
import pyspark.sql.types as typ
import os
import gc
import pyspark.sql.functions as fn
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
        df_train.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_train.columns]).show()
        print('查看测试集中每一列的缺失比例')
        df_test.agg(*[(1-(fn.count(c) /fn.count('*'))).alias(c+'_missing') for c in df_test.columns]).show()
        # allCol=df_train.columns
        # for c in allCol:
        #     num=df_train.filter(fn.isnan(c)).count()
        #     print('{}:该列存在nan值的行数为{}'.format(c,num))


        #三表关联后，有些item_id是没有title的，导致title部分数据可能存在nan值，这里要进行缺失值填充

        #类别变量填充-1，连续变量用均值填充
        #对'title_topic'这一列填充-1即可
        df_train=df_train.na.fill({'title_topic': -1})
        df_test=df_test.na.fill({'title_topic': -1})


        return df_train,df_test




    def data_process(self,df_train,df_test):
        # sqlContext = SQLContext(self.sc)
        df_train_count=df_train.count()
        print('df_train_count:19622340')     #全部记录数：22384139
        print('df_test_count:2761799')

        localPath='/data/code/DeepCTR/data/'
        train_label = df_train['finish','like']
        test_label = df_test['finish','like']
        print('保存test_label')
        test_label.toPandas().to_csv(localPath+"test_label.csv",index=False)
        print('保存test_label结束')

        df=df_train.union(df_test)
        # df.cache()  #不能用cache(),本身内存不够，再将df保存在内存中，会导致内存溢出
        feat_dict = {}
        cnt = 1
        #把duration_time列转化为VectorUDT
        df=VectorAssembler(inputCols=['duration_time',],outputCol='duration_time_feature').transform(df)
        # df.select('duration_time_feature').show(5)

        '''
        actLog：item_city,user_city,item_id,uid,channel,finish,like,duration_time,item_pub_hour,device_Cnt_bin,authorid_Cnt_bin,musicid_Cnt_bin
                uid_playCnt_bin,uid_playCnt_bin,itemid_playCnt_bin,musicid_Cnt_bin,user_city_score_bin,item_city_score_bin
        nlp_topic:item_id,title_topic
        face:item_id,gender,beauty,relative_position_0,relative_position_1,relative_position_2,relative_position_3
        '''
        ca_col = [ 'channel','item_pub_hour','device_Cnt_bin','authorid_Cnt_bin','musicid_Cnt_bin', \
                  'uid_playCnt_bin','itemid_playCnt_bin','user_city_score_bin','item_city_score_bin' ,'title_topic']
        co_col = ['duration_time_feature',]
        # co_col_undeal=['topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9',\
        #              'topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19',\
        #              'topic20','topic21','topic22','topic23','topic24','topic25','topic26','topic27','topic28','topic29',\
        #              'topic30','topic31','topic32','topic33','topic34','topic35','topic36','topic37','topic38','topic39',\
        #              'topic40','topic41','topic42','topic43','topic44','topic45','topic46','topic47','topic48','topic49',]
        co_col_undeal1=['relative_position_0','relative_position_1','relative_position_2','relative_position_3',\
                     'beauty',]   #这部分是连续变量，缺失部分已经用均值填充过了，同时其最大最小值都在0-1之间，因此不需要再做额外的处理


        #不参与训练的列 ：item_id\uid

        #对连续变量做标准化处理
        for col in co_col:
            scaler = StandardScaler(inputCol=col, outputCol="scaled"+col,    withStd=True, withMean=True)
            scalerModel = scaler.fit(df)   #duration_time must be type of VectorUDT,but was actually IntegerType
            df = scalerModel.transform(df)
            feat_dict[col] = cnt  #index
            cnt += 1
        #对列重命名
        for col in co_col:
            df=df.drop(col)
            df=df.withColumnRenamed("scaled"+col,col)

        #topic是连续变量，但是不需要做标准化处理
        for col in co_col_undeal1:
            feat_dict[col] = cnt  #index
            cnt += 1

        def to_array(col):
            def to_array_(v):
                return v.toArray().tolist()
            return fn.udf(to_array_, typ.ArrayType(typ.DoubleType()))(col)

        df=df.withColumn("duration_time", to_array(fn.col("duration_time_feature")))
        df = df.withColumn("duration_time", df["duration_time"].getItem(0))
        df=df.drop("duration_time_feature")

        df = df.withColumn("duration_time",fn.bround("duration_time", scale=2))
        #保留两位小数
        # for col  in co_col_undeal :
        #     df = df.withColumn(col,fn.bround(col, scale=2))


        #对类别变量进行处理
        for col in ca_col:
            us=df.select(col).distinct().rdd.map(lambda r: r[0]).collect()
            # print(col)
            # print(len(us))
            feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))   #类别变量col里面每个value都分配一个index
            cnt += len(us)

        # print(cnt)

        feat_dim_file = open(localPath+"feat_dim.txt",mode='w',encoding='utf-8')
        feat_dim_file.write(str(cnt))
        feat_dim_file.close()

        # print('查看df有哪些列')
        # df.printSchema()

        co_col=["duration_time",]
        feature_value=df.select(co_col+ca_col)

        #创建一个与feature_value相同的dataFrame
        feature_index = feature_value

        # print('数据列有以下这些')
        # print(feature_index.columns)

        #修改dataFrame中列的值,把下面部分的功能转化成spark中的操作
        for i in feature_index.columns:
            # print(i)
            # print(feat_dict[i])  #这是一个dict
            if i in co_col:
                #替换已经存在的同名列,返回一个新的dataframe,
                if i=="duration_time":
                    a=feat_dict["duration_time_feature"]
                else:
                    a=feat_dict[i]
                # AssertionError: col should be Column,第二个参数必须是含有列的表达式
                feature_index=feature_index.withColumn(i, fn.lit(a))
            if i in ca_col:
                k=list(feat_dict[i].keys())
                v=list(feat_dict[i].values())
                feature_index=feature_index.replace(k,v,i)  #k,v都是list，对i列中的k用v来替代
                feature_value=feature_value.withColumn(i, fn.lit(1))


        feature_index=feature_index.repartition(1).withColumn("id", monotonically_increasing_id())
        train_feature_index=feature_index.filter(feature_index['id']<df_train_count)
        test_feature_index=feature_index.filter(feature_index['id']>=df_train_count).drop('id')

        feature_value=feature_value.repartition(1).withColumn("id", monotonically_increasing_id())
        train_feature_value=feature_value.filter(feature_value['id']<df_train_count)
        test_feature_value=feature_value.filter(feature_value['id']>=df_train_count).drop('id')

        train_label = train_label.repartition(1).withColumn("id", monotonically_increasing_id())


        #再重新分区，分区数确定：executor个数*每个executor的core的个数=集群总核数,分区成更小的快传入内存
        #repartition会增加shuffle，性能不一定提高，需要尝试
        #之后toPandas操作，这里的分区决定了有多少个task来执行toPandas操作，分区成更小的快传入内存，不会导致内存溢出
        #实践中，未分区之前会导致内存溢出
        test_feature_value=test_feature_value.repartition(300)
        train_feature_value=train_feature_value.repartition(300)

        # print(train_feature_value.count())  #19622340
        # print(test_feature_value.count())   #2761799



        print('-------5.保存数据预处理结果-------')
        # path = self.parser.get("hdfs_path", "hdfs_data_path")

        # def toCSVLine(data):
        #   return ','.join(str(d) for d in data)

        # def toCSVLineFromPartition(dataSets):  #这一步执行后，转化为str，内存占用变大，导致内存溢出
        #     final_iterator=[]
        #     for subdata in dataSets:
        #         final_iterator.append(','.join(str(d) for d in subdata))
        #     return iter(final_iterator)

        #将训练集划分成10份，
        segList= [i for i in range(0,df_train_count+1,df_train_count//10)]
        # print('train存储方案一：toPandas()后保存到本地')
        for i in range(len(segList)):
            if i <=9:
                # print(i)
                # print(segList[i],segList[i+1])
                train_feature_index_i=train_feature_index.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                train_feature_value_i=train_feature_value.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                train_label_i=train_label.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')

                train_feature_index_i.toPandas().to_csv(localPath+"train_feature_index_"+str(i)+".csv",index=False)
                train_feature_value_i.toPandas().to_csv(localPath+"train_feature_value_"+str(i)+".csv",index=False)
                train_label_i.toPandas().to_csv(localPath+"train_label_"+str(i)+".csv",index=False)

        print('test存储方案一：toPandas()后保存到本地')
        test_feature_index.toPandas().to_csv(localPath+"test_feature_index.csv",index=False)
        test_feature_value.toPandas().to_csv(localPath+"test_feature_value.csv",index=False)
        # test_label.toPandas().to_csv(localPath+"test_label.csv",index=False)


        # print('test存储方案二：保存在hdfs上，用的时候再get下来')
        # test_feature_index.rdd.mapPartitions(toCSVLineFromPartition).saveAsTextFile(path+"test_feature_index_"+str(i)+".csv")
        # test_feature_value.rdd.mapPartitions(toCSVLineFromPartition).saveAsTextFile(path+"test_feature_value_"+str(i)+".csv")
        #




if __name__ == "__main__":
    spark_job = SparkFEProcess()

    df_train,df_test=spark_job.data_describe()
    # spark_job.data_process(df_train,df_test)