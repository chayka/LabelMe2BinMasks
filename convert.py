import json
import os

from glob import glob, iglob
import re
import base64
from shutil import rmtree
import argparse
import numpy as np
import cv2
import re

parser = argparse.ArgumentParser(description='Convert set of labelme files to cocodataset format')

parser.add_argument('dataset', metavar='DATASET_NAME', type=str,
                    help='dataset (folder) name')

parser.add_argument('-w, --width', dest='width', type=int,
                    help='output width', default=640)

parser.add_argument('-h, --height', dest='height', type=int,
                    help='output height', default=512)

parser.add_argument('-g, --greyscale', dest='greyscale', action='store_true',
                    help='convert images to greyscale', default=False)

parser.add_argument('-p, --label-mask-pattern', dest='label_mask_pattern',
                    help='provide label pattern to extract only specific masks', default=False)

args = parser.parse_args()

# print(args)
# exit()

INPUT_DIR = "./input/"
OUTPUT_DIR = "./output/"
DATASET_DIR = OUTPUT_DIR + "{}/".format(args.dataset)
IMAGES_DIR = DATASET_DIR + "/images"
# MASKS_DIR = DATASET_DIR + "/masks"

def empty_dir(path):
    """ empty specified dir """
    if os.path.exists(path):
        rmtree(path)
    os.mkdir(path)


def ensure_dir(path):
    """ empty specified dir """
    if not os.path.exists(path):
        os.mkdir(path)


def get_bbox(coords):
    """ get bounding box in format [tlx, tly, w, h] """
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for [x, y] in coords:
        min_x = x if not min_x else min(x, min_x)
        min_y = y if not min_y else min(y, min_y)
        max_x = x if not max_x else max(x, max_x)
        max_y = y if not max_y else max(y, max_y)

    return [min_x, min_y, max_x - min_x, max_y - min_y]


file_pattern = re.compile('([^/]*)\.([^/.]+)$')
category_pattern = re.compile('panel', re.IGNORECASE)

imageId = 0
annId = 0
categoryId = 0

ensure_dir(INPUT_DIR)
ensure_dir(OUTPUT_DIR)
empty_dir(DATASET_DIR)
ensure_dir(IMAGES_DIR)

categories = {'total': 0}

""" Browse through all marked json files """
for file in iglob(INPUT_DIR + '{}/*.json'.format(args.dataset)):
    imageId += 1
    categories['total'] += 1
    with open(file, 'r') as f:

        """ Load json files """
        data = json.load(f)

        """ Save image file """
        file_name = "{}/{:08d}.jpg".format(IMAGES_DIR, imageId)
        print(file_name, file)
        image_data = base64.b64decode(data["imageData"])
        with open(file_name, 'wb') as fi:
            fi.write(image_data)

        """ Get image width x height """
        im = cv2.imread(file_name)
        (width, height, _) = im.shape

        im = cv2.resize(im, (args.width, args.height))
        if args.greyscale:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("{}/{:08d}.tif".format(DATASET_DIR, imageId), im)
            cv2.imwrite("{}/{:08d}.tif".format(IMAGES_DIR, imageId), im)
        else:
            cv2.imwrite(file_name, im)
            # cv2.imwrite("{}/{:08d}.jpg".format(DATASET_DIR, imageId), im)

        """ Process each shape (annotation) """
        masks = {}
        for shape in data['shapes']:
            annId += 1
            label = shape['label']

            for cat in re.split("\s+", label):
                cat = str.lower(cat).replace(' ', '_')

                """ init mask if needed """
                if cat not in masks:
                    masks[cat] = np.zeros((width, height, 1), np.uint8)
                mask = masks[cat]

                """ draw a segment """
                segment = np.int32(np.array(shape['points']))
                cv2.fillPoly(mask, [segment], 255)

        for cat, mask in masks.items():
            if cat not in categories:
                categories[cat] = 0
                ensure_dir("{}/masks_{}".format(DATASET_DIR, cat))
            categories[cat] += 1
            mask = cv2.resize(mask, (args.width, args.height))
            # cv2.imwrite("{}/{:08d}_mask_{}.tif".format(DATASET_DIR, imageId, cat), mask)
            cv2.imwrite("{}/masks_{}/{:08d}.tif".format(DATASET_DIR, cat, imageId), mask)

print(categories)
