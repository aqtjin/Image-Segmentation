import redis
import cv2
import time
import helper
import settings
import uuid
import json
import numpy as np


DB = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def create_redis_pool(host, port, db):
    pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
    return pool


def enqueue_image(uri, img):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            print("You have pushed an image with path: ",
                    img, "the path is invalid, skipped.")
            return

    # force resize here to avoid input image shape inconsistent
    # if the shape is consistent, it would not affect the data
    img = helper.resize_image(img, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH)
    data = cv2.imencode(".jpg", img)[1]

    img_encoded = helper.base64_encode_image(data)

    d = {"uri": uri, "image": img_encoded}

    inf = DB.info()

    try:
        if inf['used_memory'] >= inf['maxmemory'] * settings.input_threshold:
            raise redis.exceptions.ConnectionError
        DB.xadd("image_stream", d)
        print("Write to Redis successful")
    except redis.exceptions.ConnectionError:
        print("Redis queue is full, please wait for inference "
                "or delete the unprocessed records.")
        time.sleep(settings.interval_if_error)

    except redis.exceptions.ResponseError as e:
        print(e, "Redis memory is full, please dequeue or delete.")
        time.sleep(settings.interval_if_error)


def dequeue():
    res_list = DB.keys('result:*')
    decoded = {}
    for res in res_list:
        print(res)
        res_dict = DB.hgetall(res.decode('utf-8'))
        res_id = res.decode('utf-8').split(":")[1]
        res_value = res_dict[b'value'].decode('utf-8')
        print(res_value)
        decoded[res_id] = res_value
        DB.delete(res)
    return decoded


def image_enqueue(image_path):
    with open(image_path, "rb") as imageFile:
        # generate an ID for the classification then add the
        # classification ID + image to the queue
        k = str(uuid.uuid4())
        image = helper.base64_encode_image(imageFile.read())
        d = {"id": k, "path": image_path, "image": image}
        DB.rpush(settings.IMAGE_QUEUE, json.dumps(d))


def image_enqueue_as_stream(image_path):
    with open(image_path, "rb") as imageFile:
        # generate an ID for the classification then add the
        # classification ID + image to the queue
        k = str(uuid.uuid4())
        image = helper.base64_encode_image(imageFile.read())
        d = {"id": str(k), "path": image_path, "image": image}
        DB.xadd(settings.IMAGE_STREAMING, d)


def image_dequeue():
    image_ids = []
    images = []
    batch = None
    queue = DB.lrange(settings.IMAGE_QUEUE, 0,
                      settings.BATCH_SIZE - 1)
    for record in queue:
        record = json.loads(record.decode("utf-8"))
        image = helper.base64_decode_image(record["image"])
        image = helper.byte_to_mat(image, dtype=settings.IMAGE_DTYPE)
        image = helper.resize_image(image, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, settings.IMAGE_MIN_SCALE)
        print(np.nonzero(image))
        image = helper.NHWC2HCHW(image)
        # check to see if the batch list is None
        if batch is None:
            batch = image
        # otherwise, stack the data
        else:
            batch = np.vstack([batch, image])
        images.append(image)
        image_ids.append(record["id"])
    DB.ltrim(settings.IMAGE_QUEUE, len(image_ids), -1)
    return image_ids, images, batch


def image_dequeue_as_stream():
    queue = DB.xread({settings.IMAGE_STREAMING: b"0-0"},
                     block=0,
                     count=10)

    return queue
