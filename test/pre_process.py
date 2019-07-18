import  sys
sys.path.append("..")
from actionlog_feature_job import  *

schema = [('uid', types.StringType()),
                  ('user_city', types.StringType()),
                  ('item_id', types.StringType()),
                  ('author_id', types.StringType()),
                  ('item_city', types.StringType()),
                  ('channel', types.StringType()),
                  ('finish', types.StringType()),
                  ('like', types.StringType()),
                  ('music_id', types.StringType()),
                  ('device', types.StringType()),
                  ('time', types.StringType()),
                  ('duration_time', types.IntegerType())]

schema_withIndex = [('uid', types.StringType()),
                  ('user_city', types.StringType()),
                  ('item_id', types.StringType()),
                  ('author_id', types.StringType()),
                  ('item_city', types.StringType()),
                  ('channel', types.StringType()),
                  ('finish', types.StringType()),
                  ('like', types.StringType()),
                  ('music_id', types.StringType()),
                  ('device', types.StringType()),
                  ('time', types.StringType()),
                  ('duration_time', types.IntegerType()),
                  ('index', types.IntegerType())]


'''
预处理：
添加索引列：训练集，测试集
'''

spark_job = SparkFeatureEngine()

def set_index(x):
    global  idx
    if x is not None:
        idx += 1
        return idx_list[idx - 1]

def durationTime_preprocess(df):
    #1.计算平均值
    means_value = df.select(functions.mean('duration_time')).collect()[0][0]
    #2.计算方差
    devs_value = df.select(((df.duration_time - means_value)**2).alias('deviation'))
    #3.计算标准差
    stddev_value = math.floor(math.sqrt(devs_value.groupBy().avg('deviation').first()[0]))

    #4.用均值的两倍标准差替代离群值
    # df = df.select(
    #     df.duration_time,
    #     functions.when(df.duration_time.between(0, 300), df.duration_time)
    #         .otherwise(means_value)
    #         .alias("updated_salary")
    # )
    # return  df
    df = df.select('*').filter(df.duration_time <= 300).fillna(means_value, ['duration_time'])

    return df

if __name__ == '__main__':
    file = sys.argv[1]
    train_actionLog_df = spark_job.load_train_actionLog_df(file, schema)
    train_actionLog_df_count = train_actionLog_df.count()
    idx_list = [x for x in range(1, train_actionLog_df_count + 1)]
    idx = 0
    index_udf = udf(set_index, IntegerType())
    train_actionLog_df = train_actionLog_df.coalesce(1)
    train_actionLog_df = train_actionLog_df.select(col('*'), index_udf('uid').alias('index'))
    train_actionLog_df.show()

    '''
    duration time field pre process
    '''
    train_actionLog_df = durationTime_preprocess(train_actionLog_df)

    print(train_actionLog_df.columns)
    os.system("hadoop fs -rmr {}".format('/user/hadoop/icmechallenge2019/track2/indexd_process/'))
    col_num = len(train_actionLog_df.columns)
    train_actionLog_rdd = train_actionLog_df.rdd.map(lambda x: '\t'.join([str(x[i]) for i in range(col_num)]))
    train_actionLog_rdd.coalesce(1, True).saveAsTextFile('/user/hadoop/icmechallenge2019/track2/indexd_process/')