deepfm.py
运行  python deepfm.py
关键点：由于内存限制，
       数据预处理后的，输入到deepfm.py的训练数据，是分成了10个文件
       所以代码中训练模型时，是循环读取训练数据，分批迭代，保留效果最好的模型

deepfm_spark.py
       未完待续