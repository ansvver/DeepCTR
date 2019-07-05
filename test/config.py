from pyspark.sql.types import *
import  pyspark.sql.types as types

feature_root_dir = '/user/hadoop/icmechallenge2019/track2/test/features/'

actionLog_schema = [('uid', types.StringType()),
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

actionLog_schema_withIndex = [('uid', types.StringType()),
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


music_schema = [('music_id', types.StringType()),
                ('music_finish_ratio', types.FloatType()),
                ('music_like_ratio', types.FloatType()),
                ('music_finish_ratio_bins', types.IntegerType()),
                ('music_like_ratio_bins', types.IntegerType())]

device_schema = [('device', types.StringType()),
                ('device_finish_ratio', types.FloatType()),
                ('device_like_ratio', types.FloatType()),
                ('device_finish_ratio_bins', types.IntegerType()),
                ('device_like_ratio_bins', types.IntegerType())]

item_schema = [('item_id', types.StringType()),
                ('item_finish_ratio', types.FloatType()),
                ('item_like_ratio', types.FloatType()),
                ('item_finish_ratio_bins', types.IntegerType()),
                ('item_like_ratio_bins', types.IntegerType())]

author_schema = [('author_id', types.StringType()),
                ('author_finish_ratio', types.FloatType()),
                ('author_like_ratio', types.FloatType()),
                ('author_finish_ratio_bins', types.IntegerType()),
                ('author_like_ratio_bins', types.IntegerType())]

user_schema = [('uid', types.StringType()),
                ('user_finish_ratio', types.FloatType()),
                ('user_like_ratio', types.FloatType()),
                ('user_finish_ratio_bins', types.IntegerType()),
                ('user_like_ratio_bins', types.IntegerType()),
                ('distinct_cnt_authorId_bin', types.IntegerType()),
                ('distinct_cnt_musicId_bin', types.IntegerType())]

basic_feature_schema = [('uid', types.StringType()),
                        ('item_id', types.StringType()),
                        ('author_id', types.StringType()),
                        ('music_id', types.StringType()),
                        ('device', types.StringType()),
                        ('index', types.IntegerType()),
                        ('duration_time_bins', types.IntegerType()),
                        ('user_city_encode', types.IntegerType()),
                        ('item_city_encode', types.IntegerType()),
                        ('channel_encode', types.IntegerType()),
                        ('finish', types.StringType()),
                         ('like', types.StringType())]