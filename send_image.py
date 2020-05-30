from redisop import *
import sys

def test_enqueue(pool, image_path):
    for i in range(100000):
        image_enqueue_as_stream(image_path, pool)


if __name__ == "__main__":
    print(sys.argv)
    redis_host = sys.argv[1]
    redis_port = int(sys.argv[2])
    pool = create_redis_pool(redis_host, redis_port, '0')
    image_path = sys.argv[3]
    test_enqueue(pool, image_path)
