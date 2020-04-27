from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.streaming import StreamingContext
from load_model import load_model
from helper import *
from redisop import *
import redis
import argparse
from zoo.common.nncontext import *


DB = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def process_batch(stream_batch, batch_id):
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
    result = list(map(lambda x: model.predict(x), image_set))
    for i in range(len(result)):
        print(result[i].shape)
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

    sc = init_spark_on_local()
    ssc = StreamingContext(sc, 3)

    spark = SparkSession \
        .builder \
        .appName("Streaming Image Consumer") \
        .config("spark.redis.host", settings.REDIS_HOST) \
        .config("spark.redis.port", settings.REDIS_PORT) \
        .getOrCreate()

    # Streaming schema
    imageSchema = StructType().add("id", "string").add("path", "string").add("image", "string")
    loadedDf = spark.readStream.format("redis") \
        .option("stream.keys", settings.IMAGE_STREAMING) \
        .option("stream.read.batch.size", settings.BATCH_SIZE) \
        .option("stream.read.block", settings.BLOCK) \
        .schema(imageSchema).load()
    model = load_model(args.model_path, args.weight_path, 1)
    query = loadedDf.writeStream \
        .foreachBatch(process_batch).start()

    query.awaitTermination()
    # model = load_model(args.model_path,
    #                    args.weight_path,
    #                    1)

    # Create the queue through which RDDs can be pushed to
    # a QueueInputDStream

    # process_rdd(queue)
    #
    # ssc.start()
    #
    # ssc.awaitTermination()
