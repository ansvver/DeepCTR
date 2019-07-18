#!/usr/local/bin/python

import os
import  sys
import  json

current_path = os.path.dirname(os.path.realpath(__file__))
workspace_path = current_path.split('featureEngineering')[0]
sys.path.append(os.path.join(workspace_path, 'test'))

from pyspark import SparkConf, SparkContext
from  pyspark.sql import  *
from pyspark.sql.types import *
import  configparser
import  pyspark.sql.types as types
from pyspark.sql import  functions
from pyspark.ml.feature import StandardScaler
import  json
import math

from pyspark.ml.clustering import LDA, LDAModel,DistributedLDAModel
from pyspark.ml.linalg import SparseVector


"""
计算每个单词在每个title下的tf
tf值意义：词w在文档d中出现次数count(w, d)和文档d中总词数size(d)的比值：
"""
def tf(line_json):

    cnt_map = {}
    item_id = line_json.get('item_id')
    value = line_json.get('title_features')
    title_count = 0
    for key, value in value.items():
        title_count += 1
        cnt_map[key] = cnt_map.get(key, 0) + int(value)
    return [(item_id, (w, cnt * 1.0 / title_count)) for w, cnt in cnt_map.items()]

#value: (itemId, score)
#RDD[(K, V)]  --->  RDD[(K, C)]
#V变成C，正如这里(itemId, tf_value) ---> ([(itemId, tf_value)], 1)
def create_combiner(value):
    t = []
    t.append(value)
    return (t, 1)

#merge a V into a C
#value格式:(itemId, tf)
#C的格式：([(itemId_1,tf1), .....(itemId_n, tfn)], total)
#自定义merge方法：将一个还未combine的value合并到C中
#value的值加入到C中C[0]:list里面，同时
def merge(C, V):
    """
    x==>(list, count)
    x[0] == list
    x[1] == count

    :param x:
    :param v:
    :return:
    """

    t = []
    if C[0] is not None:
        t = C[0]
    t.append(V)
    return (t, C[1] + 1)

#自定义将两个C1 C2 进行合并
#C1 = ([(itemId_1, tf1), (itemId_2, tf2)], 3)
#C2 = ([(itemId_3, tf3), (itemId_4, tf4)], 5)
#合并后： ([(itemId_1, tf1), (itemId_2, tf2), (itemId_3, tf3), (itemId_4, tf4)], 8)
def merge_combine(x, y):
    t1 = []
    t2 = []
    if x[0] is not None:
        t1 = x[0]
    if y[0] is not None:
        t2 = y[0]
    t1.extend(t2)
    return (t1, x[1] + y[1])

def flat_map_2(line, broadcast_title_count):
    rst = []
    #idf_value = line[1][-1] * 1.0 / broadcast_title_count
    idf_value =  math.log

    for doc_pair in line[1][:-1]:
        #print(doc_pair)
        for p in doc_pair:
            rst.append(Row(itemId=p[0], token=line[0], tf_value=p[1], idf_value=idf_value, tf_idf_value=p[1] * idf_value))
    return rst
#tf-idf trans
def trans(data):
    x = data
    #print(x[1][0], x[1][0], x[1][1])
    return (data[1][0], (data[0], data[1][1]))

#提取itle里面的word集合，用于后续总词表构造
def get_title_words(line_json):
    words_list = []
    feature_dict = line_json.get('title_features')
    for (key, value) in feature_dict.items():
        words_list.append(int(key))
    return words_list

#dict里面的key 为word index，将该dict 按index进行排序
def sort_dict_byKey(dict_data):
    tuple_list = sorted(dict_data.items(), key=lambda d: int(d[0]))
    dict_new = {}
    for tuple in tuple_list:
        key = tuple[0]
        value = tuple[1]
        dict_new[key] = value

    print(dict_new)
    return  dict_new

def cast_dict_str2int(data):
    new_dict = {}
    for item in data.items():
        key = int(item[0])
        value = item[1]
        new_dict[key] = value

    return  new_dict

def dict2StrLine(data):
    line_list = []
    for item in data.items():
        word = 'lda_' + item[0]
        value = item[1]
        for i in range(0, value):
            line_list.append(word)
    line_str = " ".join(line_list)
    print(line_str)
    return  line_str

class NLPFeatureSparkTask:

    def __init__(self):

        self.parser = self.init_config()

        sparkConf = SparkConf().setAppName("feature engineering on spark") \
            .set("spark.ui.showConsoleProgress", "false")
        self.sc = SparkContext(conf=sparkConf)
        self.sc.broadcast(self.parser)
        self.sqlContext = SQLContext(self.sc)
        # self.init_logger()
        #self.lda_model_path = '/user/hadoop/icmechallenge2019/user/michael/model/lda-model'
        self.lda_model_path = './lda'


    def init_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        workspace_path = current_path.split('featureEngineering')[0]
        config_file = workspace_path + 'resource/config.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file)
        return  parser

    def load_lda_model(self):
        print('start to load LDA model:')
        self.model = DistributedLDAModel.load(self.lda_model_path)
        self.lda = LDA.load(self.lda_path)
        print('finished load LDA model........')
        return  self.model

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



    def load_train_titleFeature_rdd(self, file):
        print('start to read data for nlp rdd:')
        raw_rdd = self.read_rdd(file)
        json_rdd = raw_rdd.map(lambda line : json.loads(line))

        total = json_rdd.count()
        print('totla count: ' + str(total))

        return json_rdd, total


    def titleFeature_data_describe(self, file):
        json_rdd, count = self.load_train_titleFeature_rdd(file)
        broadcast_title_count = self.sc.broadcast(count)
        self.tfidf_feature(json_rdd, broadcast_title_count)

    def tfidf_feature(self, title_json_rdd, broadcast_title_count):
        # [('1', ('来到', 0.3)), ('1', ('北京', 0.3))
        # [(itemId, (word, score))]
        tf_rdd = title_json_rdd.map(lambda line: tf(line))

        tf_list = tf_rdd.collect()
        print(len(tf_list))

        list = []
        for i in range(0, len(tf_list)):
            list.extend(tf_list[i])

        tf_rdd = self.sc.parallelize(list)
        idf_rdd = tf_rdd.map(lambda x: trans(x))\
                        .combineByKey(create_combiner,
                                      merge,
                                      merge_combine)
        # idf_list = idf_rdd.collect()
        # print('idf list:')
        #
        # for i in range(0, 5):
        #     print(idf_list[i])

        tf_idf_rdd = idf_rdd.flatMap(lambda line : flat_map_2(line, broadcast_title_count.value))
        tf_idf_df = self.sqlContext.createDataFrame(tf_idf_rdd)
        # tf_idf_list = tf_idf_rdd.collect()
        # print('tf-idf list:')
        # for i in range(0, 5):
        #     print(tf_idf_list[i])
        tf_idf_df.show(5)
        tf_idf_df.printSchema()

    #df = spark.createDataFrame([[1, Vectors.dense([0.0, 1.0])],[2, SparseVector(2, {0: 1.0})]], ["id", "features"])
    def lda_train(self, file):
        json_rdd, count = self.load_train_titleFeature_rdd(file)
        vocabulary_set = json_rdd.map(lambda line : get_title_words(line))\
                                 .flatMap(lambda word : word).distinct().collect()

        vocab_size = self.sc.broadcast(max(vocabulary_set) + 1)

        print('vocabulart size: ' + str(vocab_size.value))

        sparseVec_rdd = json_rdd.map(lambda line : cast_dict_str2int(line.get('title_features')))\
                                .map(lambda value : SparseVector(vocab_size.value, value))
        zip_rdd = sparseVec_rdd.zipWithIndex()
        lda_train_rdd = zip_rdd.map(lambda x : [x[1], x[0]]).cache()

        K = 4
        max_iter = 10
        seed = 1024


        lda_train_df = self.sqlContext.createDataFrame(lda_train_rdd.collect(), ["id", "features"])
        lda = LDA(k=K, maxIter = max_iter,seed=seed)
        lda_model = lda.fit(lda_train_df)

        print('LDA model vocabSize : ' + str(lda_model.vocabSize()))
        print(lda_model.isDistributed())
        lda_model.describeTopics().show()

        #os.system("hadoop fs -rmr {}".format(self.lda_model_path))
        #os.system("hadoop fs -rmr {}".format(self.lda_path))

        lda_model.write().overwrite().save(self.lda_model_path)

        self.sc.stop()

    def lda_predict_function(self, model,  dataSet):
        self.load_lda_model()
        print(dataSet)
        ll = model.logLikelihood(dataSet)
        lp = model.logPerplexity(dataSet)
        print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
        print("The upper bound on perplexity: " + str(lp))
        print('********************')



    def lda_predict_batch(self, file):

        json_rdd, count = self.load_train_titleFeature_rdd(file)
        vocabulary_set = json_rdd.map(lambda line: get_title_words(line)) \
            .flatMap(lambda word: word).distinct().collect()

        vocab_size = self.sc.broadcast(len(vocabulary_set))

        sparseVec_rdd = json_rdd.map(lambda line : cast_dict_str2int(line.get('title_features')))\
                                .map(lambda value: SparseVector(vocab_size.value, value))
        lda_train_rdd = sparseVec_rdd.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

        lda_model = self.load_lda_model()
        lda_broadcast = self.sc.broadcast(lda_model)

        lda_train_rdd.map(lambda x : self.lda_predict_function(lda_broadcast.value, x))

    def lda_input_trans(self, file):
        json_rdd, count = self.load_train_titleFeature_rdd(file)
        line_rdd = json_rdd.map(lambda line: line.get('title_features')) \
                            .map(lambda line : dict2StrLine(line))
        words_path = '/user/hadoop/icmechallenge2019/user/michael/model/words'
        os.system("hadoop fs -rm {}".format(words_path))
        line_rdd.repartition(1).saveAsTextFile(words_path)

if __name__ == "__main__":
    train_data_sample = sys.argv[1]

    spark_job = NLPFeatureSparkTask()
    #spark_job.titleFeature_data_describe(train_data_sample)

    spark_job.lda_input_trans(train_data_sample)

    #spark_job.lda_train(train_data_sample)
    #spark_job.lda_predict_batch(train_data_sample)