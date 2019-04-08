import  tensorflow as tf

from Tensorflow.build_data import  load_data

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

data = load_data()