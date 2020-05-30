from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.streaming import StreamingContext
from load_model import load_model
from helper import *
from redisop import *
import redis
import argparse
from zoo.common.nncontext import *
from visualize import *
import time


DB = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def process_batch(stream_batch, batch_id):
    t_1 = time.time()
    stream_batch = stream_batch.collect()
    image_set = []
    image_ids = []
    for q in stream_batch:
        # deserialize the object and obtain the input image
        image = base64_decode_image(q["image"])
        image = byte_to_mat(image, dtype=settings.IMAGE_DTYPE)
        image = resize_image(image, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, settings.IMAGE_MIN_SCALE)
        image = NHWC2HCHW(image)
        image_set.append(image)

        # update the list of image IDs
        image_ids.append(q["id"])

    def inference(image):
        start = time.time()
        model.predict(image)
        end = time.time()
        print("Inference time is " + str(end-start) + " ms")
    result = list(map(inference, image_set))
    for i in range(len(result)):
        enqueue(pool, str(result[i]))
    t_2 = time.time()
    print("Used time for Spark " + str(t_2-t_1) + " ms")
    # rdd = ssc.sparkContext.parallelize(image_queue[0][1])
    #
    # images = rdd.map(process_queue).collect()
    # result = list(map(lambda x: model.predict(x), images))

    # for r in range(len(queue[0][1])):
    #     image = queue[0][1][r][1][b'image'].decode("utf-8")
    #     path = queue[0][1][r][1][b'path'].decode("utf-8")
    #     image = base64_decode_image(image)
    #     image = byte_to_mat(image, dtype=settings.IMAGE_DTYPE)
    #     image = resize_image(image, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, settings.IMAGE_MIN_SCALE)
    #     image = NHWC2HCHW(image)
    #     image_set.append(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="model_path")
    parser.add_argument('--weight_path', help="weight_path")
    args = parser.parse_args()

    # sc = init_spark_on_local()
    sc = init_spark_on_yarn(
        hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop",
        conda_name="han",
        num_executor=2,
        executor_cores=10,
        executor_memory="100g",
        driver_memory="10g",
        driver_cores=4,
        extra_python_lib="/opt/work/client/pycharm_project_895/load_model.py, /opt/work/client/pycharm_project_895/redisop.py, /opt/work/client/pycharm_project_895/helper.py, /opt/work/client/pycharm_project_895/settings",
        jars="/opt/work/client/han/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-jar-with-dependencies.jar,/opt/work/client/han/spark-redis/target/spark-redis_2.11-2.4.3-SNAPSHOT-jar-with-dependencies.jar",
        spark_conf={"spark.redis.host": "172.16.0.120", "spark.redis.port": "6379"}
    )
    redis_host = sc._conf.get("spark.redis.host")
    redis_port = int(sc._conf.get("spark.redis.port"))
    pool = create_redis_pool(redis_host, redis_port, '0')

    spark = SparkSession \
        .builder \
        .appName("Streaming Image Consumer") \
        .getOrCreate()

    # Streaming schema
    imageSchema = StructType().add("id", "string").add("path", "string").add("image", "string")
    spark.sparkContext.addFile(args.model_path)
    spark.sparkContext.addFile(args.weight_path)
    loadedDf = spark.readStream.format("redis") \
        .option("stream.keys", settings.IMAGE_STREAMING) \
        .option("stream.read.batch.size", settings.BATCH_SIZE) \
        .option("stream.read.block", settings.BLOCK) \
        .schema(imageSchema).load()
    model = load_model(args.model_path, args.weight_path, 1)
    query = loadedDf.writeStream \
        .foreachBatch(process_batch).start()

    query.awaitTermination()

