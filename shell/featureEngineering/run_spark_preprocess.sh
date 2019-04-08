
source /etc/profile


echo "start to run spark job: feature engineering task"
spark-submit --master yarn \
             --driver-memory 8G  \
             --num-executors 20 \
             --executor-memory 2G \
             --executor-cores 1 \
             --deploy-mode client \
 	    /data/code/DeepCTR/featureEngineering/spark/preprocess_spark.py \
&& echo "finish spark job: feature engineering task"
