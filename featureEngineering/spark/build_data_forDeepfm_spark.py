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


        localPath='/data/code/DeepCTR/data/dataForSkearn/'
        #可能存在的问题，内存溢出
        # df_train.toPandas().to_csv(localPath+"train.csv",index=False)
        # df_test.toPandas().to_csv(localPath+"test.csv",index=False)


        return df_train,df_test




    def build_data(self):
        sqlContext = SQLContext(self.sc)
        rootPath=self.parser.get("hdfs_path", "hdfs_data_path")
        print('start to read df_test')
        test_file_path = rootPath + 'df_test'
        actLog_test_rdd = self.sc.pickleFile(test_file_path)


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

        print('start to read actLog_train_bin')
        train_file_path = rootPath + 'df_train'
        actLog_train_rdd = self.sc.pickleFile(train_file_path)
        df_train = sqlContext.createDataFrame(actLog_train_rdd,actionLogSchema)

        df_train.show(5,truncate=False)


        df_train_count=df_train.count()
        print(df_train_count)
        print('df_train_count:19622340')     #全部记录数：22384139
        print('df_test_count:2761799')

        localPath='/data/code/DeepCTR/data/dataForDeepfmTest611/'
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
        #df=VectorAssembler(inputCols=['duration_time',],outputCol='duration_time_feature').transform(df)
        # df.select('duration_time_feature').show(5)

        ca_col = [ 'channel','item_pub_hour','device_Cnt_bin','authorid_Cnt_bin','musicid_Cnt_bin', \
                  'uid_playCnt_bin','itemid_playCnt_bin','user_city_score_bin','item_city_score_bin' ,'title_topic','gender']
        co_col=['duration_time','relative_position_0','relative_position_1','relative_position_2','relative_position_3',\
                     'beauty',]   #这部分是连续变量，缺失部分已经用均值填充过了，同时其最大最小值都在0-1之间，因此不需要再做额外的处理

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


        #方案二
        idx = 0
        def add_index(df):
            print(df.count())
            index_list = [x for x in range(1, df.count()+1)]  # 构造一个列表存储索引值，用生成器会出错
            # 定义一个函数
            def set_index(x):
                global idx    # 将idx设置为全局变量
                if x is not None:
                    idx += 1
                    return index_list[idx-1]
            index = udf(set_index, typ.IntegerType())    # udf的注册，这里需要定义其返回值类型
            df1=df.select(fn.col("*"), index("channel").alias("id"))  # udf的注册的使用，alias方法用于修改列名,报错TypeError: 'str' object is not callable
            #df1.show()
            return df1

        feature_index_train=add_index(feature_index_train)
        feature_value_train=add_index(feature_value_train)
        print("观察add_index")
        feature_index_train.show(5,truncate=False)
        feature_value_train.show(5,truncate=False)
        #由于feature_index和feature_value的字段名都是一样的，所以需要对feature_value的字段名修改一下，以示区别
        train_label=add_index(train_label)

        df_concate=feature_index_train.join(feature_value_train,'id','left') \
                    .join(train_label,'id','left')
        print("各字段的区别")
        df_concate.show(5,truncate=False)


        print('-------.开始保存训练数据-------')

        #将训练集划分成10份，
        segList= [i for i in range(0,df_train_count+1,df_train_count//10)]
        # print('train存储方案一：toPandas()后保存到本地')
        for i in range(len(segList)):
            if i <=9:
                df_concate_i=df_concate.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                df_concate_i.toPandas().to_csv(localPath+"df_concate_"+str(i)+".csv",index=False)   #17+17+2
                # train_feature_index_i=feature_index_train.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                # train_feature_value_i=feature_value_train.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                # train_label_i=train_label.filter("id >={} and id <{}".format(segList[i],segList[i+1])).drop('id')
                #
                # train_feature_index_i.toPandas().to_csv(localPath+"train_feature_index_"+str(i)+".csv",index=False)
                # train_feature_value_i.toPandas().to_csv(localPath+"train_feature_value_"+str(i)+".csv",index=False)
                # train_label_i.toPandas().to_csv(localPath+"train_label_"+str(i)+".csv",index=False)
        print("训练数据保存结束")



        print("开始处理测试数据")
        feature_value_test=df.select(co_col+ca_col)
        #创建一个与feature_value相同的dataFrame
        feature_index_test = feature_value_test

        #修改dataFrame中列的值,把下面部分的功能转化成spark中的操作
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

        #划分训练集和测试集，这里已经导致无法正确划分训练集和测试集了
        #？？？？？？如何区分训练集和测试集？？？？？？？？
        #方案：在repartition(1)之前就划分好训练集和测试集，再repartition(1)---yes
        #feature_index有很多分区
        ''''
        print("方案一")
        print("取前100条数据观察")
        print("index和value是一一对应的")

        feature_index_train.limit(20).show(100)
        feature_value_train.limit(20).show(100)
        print("index和value还是一一对应的吗？")
        feature_index_train.coalesce(1).limit(20).show(100)
        feature_value_train.coalesce(1).limit(20).show(100)

        train_label.limit(100).show(100)
        train_label.coalesce(1).limit(100).show(100)


        print("添加自增id,pyspark中只能先coalesce(1)再使用自增id，否则就是每个分区有自己的自增id")
        feature_index_train=feature_index_train.coalesce(1).withColumn("id", monotonically_increasing_id())
        feature_value_train=feature_value_train.coalesce(1).withColumn("id", monotonically_increasing_id())
        train_label = train_label.coalesce(1).withColumn("id", monotonically_increasing_id())
        '''




        print("一次性toPandas()到本地")
        feature_index_train.toPandas().to_csv(localPath+"train_feature_index.csv",index=False)
        feature_value_train.toPandas().to_csv(localPath+"train_feature_value.csv",index=False)
        train_label.toPandas().to_csv(localPath+"train_label.csv",index=False)



if __name__ == "__main__":
    spark_job = SparkFEProcess()

    # df_train,df_test=spark_job.data_describe()

    spark_job.build_data()    #完成读取数据