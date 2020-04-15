import argparse
import redisop
from os import listdir
from os.path import isfile, join


def images_enqueue(dir_path):
    for f in listdir(dir_path):
        if isfile(join(dir_path, f)):
            redisop.image_enqueue_as_stream(join(dir_path, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help="Path where the images are stored")
    args = parser.parse_args()
    images_enqueue(args.img_path)
