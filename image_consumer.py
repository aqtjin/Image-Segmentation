from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from load_model import load_model
from helper import *
from redisop import *
import argparse
from zoo.common.nncontext import *
import os
os.environ["PYSPARK_PYTHON"] = "/home/aqtjin/ENTER/envs/t_env/bin/python3"
import matplotlib.pyplot as plt


DB = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


    # model = load_model(model_path, weight_path, 1)
    # images.foreach(lambda x: model.predict(x))
    # for r in range(len(queue[0][1])):
    #     image = queue[0][1][r][1][b'image'].decode("utf-8")
    #     path = queue[0][1][r][1][b'path'].decode("utf-8")
    #     image = base64_decode_image(image)
    #     image = byte_to_mat(image, dtype=settings.IMAGE_DTYPE)
    #     image = resize_image(image, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, settings.IMAGE_MIN_SCALE)
    #     image = NHWC2HCHW(image)
    #     image_set.append(image)

    # result = list(map(model.predict, images))
    # def fun(x):
    #     result = []
    #     print(type(x))
    #     print(x.shape)
    #     result.append(np.array(model.predict(x)))
    #     return result
    # mapped_stream = rdd.map(fun)
    # # m = mapped_stream.collect()
    # # def f(x):
    # #     print(x)
    # mapped_stream.foreach(print)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="model_path")
    parser.add_argument('--weight_path', help="weight_path")
    args = parser.parse_args()

    sc = init_nncontext("Streaming Image Segmentation Example")
    ssc = StreamingContext(sc, 3)
    # sc = init_spark_on_yarn(
    #     hadoop_conf="",
    #     conda_name="",
    #     num_executor=2,
    #     executor_cores=10,
    #     executor_memory="100g",
    #     driver_memory="10g",
    #     driver_cores=4,
    # )

    # Fetch images from redis
    queue = image_dequeue_as_stream()

    model = load_model(args.model_path,
                       args.weight_path,
                       1)
    # Create the queue through which RDDs can be pushed to
    # a QueueInputDStream

    def process_rdd(image_queue):
        rdd = ssc.sparkContext.parallelize(image_queue[0][1])

        def process_queue(x):
            image = x[1][b'image'].decode("utf-8")
            image = base64_decode_image(image)
            image = byte_to_mat(image, dtype=settings.IMAGE_DTYPE)
            image = resize_image(image, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, settings.IMAGE_MIN_SCALE)
            image = NHWC2HCHW(image)
            return image
        images = rdd.map(process_queue).collect()
        result = map(lambda x: model.predict(x), images)
        return result
    result = process_rdd(queue)

    ssc.start()

    ssc.awaitTermination()
