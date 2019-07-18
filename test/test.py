# import  tensorflow as tf
#
# from Tensorflow.build_data import  load_data
# import  json
#
# from sklearn.datasets import fetch_20newsgroups
#
# from pyspark.ml.clustering import LDA
# a = tf.constant([[2], [4]])
# b = tf.constant([[2,2,2,2], [3, 3,3,3]])
#
# y = tf.multiply(a, b)
# #z = tf.reduce_sum(y, 1)
#
# print("start to run")
# with tf.Session() as session:
#     print(session.run(y))
#     #print(session.run(z))

# data = load_data()

# data = '{"item_id": 4036886, "title_features": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1}}'
# print(str(data))
#
# data_json = json.loads(data)
# value  = data_json.get('title_features')
# for key,value in value.items():
#     print(key, value)


dict_demo = {"416": 1, "417": 1, "418": 1, "419": 1, "4": 1, "421": 1, "207": 1, "176": 1, "117" : 1, "420": 1, "303": 1, "60": 1}

# print(dict_demo["304"])
#
# print(dict_demo["140"])

# tuple_list = sorted(dict_demo.items(), key=lambda d: int(d[0]))
#
# print(tuple_list)
#
# dict_new = {}
# for tuple in tuple_list:
#     key = tuple[0]
#     value = tuple[1]
#     dict_new[key] = value
#
# print(dict_new)
# tuple_list = []
# # for item in dict_demo.items():
# #     print(item)
# #     tuple_list.append((int(item[0]), item[1]))
# #
# # print(tuple_list)
# #
# # dict_demo2 = {1: 1000, 2: 1, 3: 1, 4: 1}
# # print(dict_demo2[1])
# # for item in dict_demo2.items():
# #     print(item)


# test_list = []
# #
# # test_dict = {'1' : 1, '2' : 3, '4': 1}
# # for item in test_dict.items():
# #     word = 'lda_' + item[0]
# #     value = item[1]
# #     for i in range(0, value):
# #         test_list.append(word)
# #
# # test_str = " ".join(test_list)
# # print(test_str)

import  numpy as np
import random as rd
# data=[rd.randint(a=100,b=1000) for _ in range(20)]
# bins=[200,300,400,500,600,700,800,900,1000]
# print('data:',data)
# print('bins:',bins)
#
# print('np.digitize(data,bins):', np.digitize(data,bins))
#
# size = len(bins)
# start = 0
# end = size - 1
#
# def binary_search(x, bins, start, end):
#     while start <= end:
#         if x <= bins[start]:
#             return  start
#         if x >= bins[end]:
#             return end
#
#         middle = (start + end) // 2
#         if x >= bins[middle]:
#             start = middle + 1
#         elif x < bins[middle]:
#             end = end - 1
# binary = []
# # for x in data:
# #     start = 0
# #     end = len(bins) - 1
# #     binary.append(binary_search(x, bins, start, end))
# #
# #
# # print(binary)
# # print(binary[0])
# # print(binary[1])

# percents = [0, 20,25, 50, 70,90, 100]
# a = np.array([4.2, 7, 1, 3.2, 5, 13, 6.7, 10, 18])
# b = np.array([1, 3.2, 4.2, 5, 6.7, 7, 10, 13, 18])
# print(a)
# bins = []
# for percent in percents:
#     split = np.percentile(a, percent)
#     bins.append(split)
#
# print(bins)

# def eval_test(dict, key):
#     print(dict[key])
#
#
# key = 'name'
# value = 1000
#
# dict = {}
# dict['name'] = value
#
# eval_test(dict, key)

import  config

# basic_feature_list = []
# for item in config.basic_feature_schema:
#     basic_feature_list.append(item[0])
# print(basic_feature_list)

def read_train_data(file_path, field_size):
    result = {'label': [], 'index': [], 'value': [], 'feature_sizes':[]}
    cate_dict = []
    for i in range(field_size):
        print (i)
        cate_dict.append(set())
    f = open(file_path, 'r')
    for line in f:
        datas = line.strip().split('\t')
        result['label'].append(int(datas[0]))
        indexs = [int(item) for item in datas[1:]]
        values = [1 for i in range(field_size)]

        for i in range(1, field_size + 1):
            j = i - 1
            print ("j:" + str(j))
            cate_dict[j].add(int(datas[i]))

        result['index'].append(indexs)
        result['value'].append(values)

    for item in cate_dict:
        print (item)
        max_index = max(item)

        result['feature_sizes'].append(max_index + 1)  #每个field的feature value 数量

    print (result['feature_sizes'])

file_path = 'C:\conf_test\deepctr\data\deepfm-train-data'
field_size = 16
read_train_data(file_path, field_size)

