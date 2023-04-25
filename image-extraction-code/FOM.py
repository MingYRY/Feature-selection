import cv2
import numpy as np
import os
import mahotas as mt
import csv
import math, os, sys, statistics
from PIL import Image
POSITIVE_TRAIN_DIR = 'data/train_images/positives'
NEGATIVE_TRAIN_DIR = 'data/train_images/negatives'


def multiple_img_features(img):
    v, features = [], []
    maxHistogram = 256
    hist = [0] * maxHistogram
    Hg = 0
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            r = img.getpixel((i, j))
            v.append(r)
            hist[r] += 1
            if (r > Hg):
                Hg = r
    features.append(statistics.mean(v))  # Arithmetic mean (“average”) of data.
    features.append(statistics.mode(v))  # Single mode (most common value) of discrete or nominal data.
    features.append(statistics.pvariance(v))  # Population variance of data.
    features.append(statistics.pstdev(v))  # Population standard deviation of data.
    features.append(statistics.variance(v))  # Sample variance of data.
    features.append(statistics.stdev(v))  # Sample standard deviation of data
    entropy, energy = 0, 0
    for i in range(Hg):
        energy += (hist[i] ** 2)
        if (hist[i] != 0):
            entropy += -hist[i] * math.log2(hist[i])
    features.append(energy)  # Energy
    features.append(entropy)  # Entropy
    # features.append(statistics.median_high(v)) # High median of data.
    # features.append(statistics.median_low(v)) # Low median of data.
    return features


def get_image_feature_vector(image, positive=None):
    BIC_results = multiple_img_features(image)

    if positive is None:
        feature_set = BIC_results
    else:
        pos_num = 1 if positive else 0
        BIC_results = np.append(BIC_results, pos_num)
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
def create_csv_output(filename, posdir, negdir):
    # Load images
    positive_images = load_images_from_folder(posdir)
    negative_images = load_images_from_folder(negdir)

    # Get feature vectors of images
    positive_feature_vectors = get_all_image_feature_vectors(positive_images, True)
    negative_feature_vectors = get_all_image_feature_vectors(negative_images, False)

    # Input feature vectors into a CSV file
    with open(filename, "w", encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerows(positive_feature_vectors)
        writer.writerows(negative_feature_vectors)


if __name__ == '__main__':
    create_csv_output("data/train_FOM.csv", POSITIVE_TRAIN_DIR, NEGATIVE_TRAIN_DIR)
