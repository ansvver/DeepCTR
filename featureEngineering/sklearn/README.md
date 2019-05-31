sklearn版lightgbm

lgbm_params.py
调参过程
before_Step1. 学习率和估计器及其数目
先把学习率先定一个较高的值，这里取 learning_rate = 0.1，
其次确定估计器boosting_type的类型，选gbdt。
为了确定估计器的数目，也就是boosting迭代的次数，也可以说是残差树的数目，参数名为n_estimators/num_iterations/num_round/num_boost_round。
先将该参数设成一个较大的数，然后在cv结果中查看最优的迭代次数

step1:max_depth 和 num_leaves
这是提高精确度的最重要的参数。
max_depth ：设置树深度，深度越大可能过拟合
num_leaves：因为 LightGBM 使用的是 leaf-wise 的算法，因此在调节树的复杂程度时，使用的是 num_leaves 而不是 max_depth。大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。
我们也可以同时调节这两个参数，对于这两个参数调优，我们先粗调，再细调：
这里我们引入sklearn里的GridSearchCV()函数进行搜索。不知道怎的，这个函数特别耗内存，特别耗时间，特别耗精力。

Step2: min_data_in_leaf 和 min_sum_hessian_in_leaf
说到这里，就该降低过拟合了。
min_data_in_leaf 是一个很重要的参数, 也叫min_child_samples，它的值取决于训练数据的样本个树和num_leaves. 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合。
min_sum_hessian_in_leaf：也叫min_child_weight，使一个结点分裂的最小海森值之和，真拗口（Minimum sum of hessians in one leaf to allow a split. Higher values potentially decrease overfitting）

Step3: feature_fraction 和 bagging_fraction
这两个参数都是为了降低过拟合的。
feature_fraction参数来进行特征的子抽样。这个参数可以用来防止过拟合及提高训练速度。
bagging_fraction+bagging_freq参数必须同时设置，bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging。

上述调参过程存在问题是内存不足
另外，在上述调参过程中需要考虑到各树的叶子节点编号转化到lr的输入的过程中，数据矩阵会扩大num_leaves倍，
所以一开始调这个参数的时候就要考虑到内存的要求，在不大牺牲准确度的情况下，减小num_leaves


lgbm.py
经过lgbm_params.py调参之后，已经确定了最优的参数，将最优参数best_params代入，经过训练得到各树的叶子节点编号，进而转化为lr的输入，
最终衡量指标为交叉熵和auc,
目前模型训练参数为内存允许情况下的随机参数，auc大约在0.7左右


