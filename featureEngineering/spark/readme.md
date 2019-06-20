特征工程：
explore_spark_step1_single.py:
1、测试集中有以下字段缺失，并填充相应值
uid_count_bin：因为测试集中出现了新的uid，这个时候bin填充为-1
item_id_count_bin： 因为测试集中出现了新的item_id ,缺失率为0.6，所以对于新的item_id其count特征不能知道，但是后面可通过item_id可以关联nlp、face特征
author_id_count_bin：缺失率为0.14
#duration_time是一个连续变量，不应该在计算count后分箱，而是画图后直接分箱
item_city_count_bin：
music_id_count_bin：
device_count_bin：缺失率为0.03
此外，每一个bin变量对应一个ratio变量，表示对每个类别的偏好，缺失值填充为0，下同

explore_spark_step1_cross.py:
1、测试集中有以下字段缺失，并填充相应值
uid_item_id_count_bin：缺失率为0.995 ，因此判断该变量没有区分度，不构造此变量
uid_author_id_count_bin：缺失率为0.86，最终还是保留了该变量，因为考虑到实际中用户对发布者还是存在一定的偏爱
uid_item_city_count_bin：缺失率为0.18，对缺失值填充-1
uid_channel_count_bin：缺失率为0.039，对缺失值填充-1
uid_music_id_count_bin：缺失率为0.256，对缺失值填充-1
uid_device_count_bin：缺失率为0.03，对缺失值填充-1
author_id_channel_count_bin：缺失率为0.18，对缺失值填充-1
author_id_user_city_count_bin：缺失率为0.46，对缺失值填充-1
author_id_item_city_count_bin：缺失率为0.17，对缺失值填充-1
author_id_music_id_count_bin：缺失率为0.28，对缺失值填充-1
uid_channel_device_count_bin：缺失率为0.05，对缺失值填充-1
author_id_music_id_item_pub_hour_count_bin：缺失率为0.35，对缺失值填充-1


explore_spark_step2.py:读取上述两步的处理结果，针对uid，authorid，musicid等组合，统计各组合特征的正样本概率
具体特征有：



