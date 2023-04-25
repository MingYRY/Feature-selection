import cv2
import numpy as np
import os
import mahotas as mt
import csv
from PIL import Image
ZERO_TRAIN_DIR = 'data/train_images/zero'
ONE_TRAIN_DIR = 'data/train_images/one'
TWO_TRAIN_DIR = 'data/train_images/two'


def multiple_img_features(img):
    factor = 64
    HI, HE = [0] * factor, [0] * factor
    N,M = img.size
    for x in range(N):
        for y in range(M):
            ind = int((img.getpixel((x, y)) / 255) * (factor - 1))
            if (x == 0 or y == 0 or x + 1 == N or y + 1 == M):
                HE[ind] += 1
            else:
                if (img.getpixel((x, y)) == img.getpixel((x, y - 1)) and
                        img.getpixel((x, y)) == img.getpixel((x, y + 1)) and
                        img.getpixel((x, y)) == img.getpixel((x - 1, y)) and
                        img.getpixel((x, y)) == img.getpixel((x + 1, y))):
                    HI[ind] += 1
                else:
                    HE[ind] += 1
    return HI + HE


def get_image_feature_vector(image, positive):
    BIC_results = multiple_img_features(image)

    if positive == 0:
        BIC_results = np.append(BIC_results, 0)
        feature_set = BIC_results
    if positive == 1:
        BIC_results = np.append(BIC_results, 1)
        feature_set = BIC_results

    return feature_set


# Get all feature vectors of a set of images, used when building the CSV files
def get_all_image_feature_vectors(images, positive):
    feature_sets = []

    for image in images:
        feature_set = get_image_feature_vector(image, positive)
        feature_sets.append(feature_set)

    return feature_sets


# 返回给定目录中的图像列表
# Return a list of images from the given directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('RGB').convert('L')
        if img is not None:
            images.append(img)
    return images


# 路径信息
def create_csv_output(filename, zerodir, onedir):
    # Load images
    zero_images = load_images_from_folder(zerodir)
    one_images = load_images_from_folder(onedir)

    # Get feature vectors of images
    zero_feature_vectors = get_all_image_feature_vectors(zero_images, 0)
    one_feature_vectors = get_all_image_feature_vectors(one_images, 1)

    # Input feature vectors into a CSV file
    with open(filename, "w", encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerows(zero_feature_vectors)
        writer.writerows(one_feature_vectors)


if __name__ == '__main__':
    create_csv_output("data/train_BIC.csv", ZERO_TRAIN_DIR, ONE_TRAIN_DIR)
