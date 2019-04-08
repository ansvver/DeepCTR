dos2unix工具：windows下编辑过的文件在linux上可能会存在^M 符号，
             在linux上直接调用dos2unix filename即可消除末尾的换行符号

#pytorch dnn_ctr 参考链接
The framework to deal with ctr problem

details: https://zhuanlan.zhihu.com/p/32885978

FNN's introduction and api: https://zhuanlan.zhihu.com/p/33045184

PNN's introduction and api: https://zhuanlan.zhihu.com/p/33177517

DeepFM's introduction and api: https://zhuanlan.zhihu.com/p/33479030

AFM's introduction and api: https://zhuanlan.zhihu.com/p/33540686

NFM's introduction and api: https://zhuanlan.zhihu.com/p/33587540

DCN's introduction and api: https://zhuanlan.zhihu.com/p/33619389

短视频原始数据地址：
local: zb2620  /data/code/douyinData
hdfs: http://10.8.26.21:50070/explorer.html#/user/hadoop/icmechallenge2019

短视频/广告特征工程参考：
https://zhuanlan.zhihu.com/p/38341881
1. 对于uid这类特别稀疏的原始特征，不需要直接进行one-hot编码而导致特征维度急剧膨胀，
   可以考虑通过对uid进行出现的样本数量统计，或者使用uid的转化率或者正样本数来对uid来进行表示
2. 两两特征组合后，如果稀疏程度较高，也通过统计特征来进行压缩


推荐系统特征类型分类：

1. 用户与视频的基本属性特征
    1.1.类别变量
    category_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'device', ]
    高稀疏变量： uid, item_id, author_id, music_id, device
    一般稀疏变量: user_city, item_city, channel

    1.2.连续变量
    continuous_features = ['duration_time']  # 作品时长

2. 视频内容特征：
    2.1 NLP title特征： (word， frequence)
    2.2 视频人脸特征： gender， beauty， relative_position
    2.3 音频特征： embedding向量，128维
    2.4 视觉特征：