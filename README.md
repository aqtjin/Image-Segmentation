# Image-Segmentation
Based on Analytics-zoo and Spark-Streaming, do image segmentation.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via pip or download the prebuilt package.
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

## Prepare models
See [here](https://docs.openvinotoolkit.org/latest/index.html) to install OpenVINO™ Toolkit, and follow the instructions to download the pre-trained models and transform the models to IR(including .xml and .bin).

## Run with prebuilt package
```
export JAVA_HOME=/opt/jdk1.8.0_65
export CLASSPATH=.:${JAVA_HOME}/lib:${JAVA_HOME}/jre/lib:${JAVA_HOME}/lib/tools.jar:${JAVA_HOME}/lib/dt.jar
export PATH=${JAVA_HOME}/bin/:${JAVA_HOME}/jre/bin:${PATH}
export SPARK_HOME=/opt/work/client/han/spark-2.4.3-bin-hadoop2.7
export HADOOP_HOME=/opt/work/hadoop-2.7.2
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop

$SPARK_HOME/bin/spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./py36/bin/python \
    --master yarn \
    --deploy-mode client \
    --driver-memory 10g \
    --executor-memory 100g \
    --executor-cores 10 \
    --num-executors 2 \
    --jars /opt/work/client/han/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-jar-with-dependencies.jar,/opt/work/client/han/spark-redis/target/spark-redis_2.11-2.4.3-SNAPSHOT-jar-with-dependencies.jar \
    --conf "spark.redis.host=172.16.0.120" \
    --conf "spark.redis.port=6379" \
    /opt/work/client/han/pycharm_project_895/image_consumer.py \
    --model_path "/opt/work/client/han/frozen_inference_graph.xml" \
    --weight_path "/opt/work/client/han/frozen_inference_graph.bin"
```
