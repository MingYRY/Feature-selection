import cv2
import numpy as np
import os
import mahotas as mt
import csv

POSITIVE_TRAIN_DIR = 'data/train_images/positives'
NEGATIVE_TRAIN_DIR = 'data/train_images/negatives'


def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    features = mt.features.tas(np.asanyarray(image))

    # take the mean of it and return it
    return features


def multiple_img_features(image):
    """
    Input - A directory containing only all images
    Output - A dictionary with 13 features
    """
    Haral = extract_features(image)

    return Haral


def get_image_feature_vector(image, positive=None):
    Haralick_results = multiple_img_features(image)

    if positive is None:
        feature_set = Haralick_results
    else:
        pos_num = 1 if positive else 0
        Haralick_results = np.append(Haralick_results, pos_num)
        feature_set = Haralick_results

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
        img = cv2.imread(os.path.join(folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (13, 13), 0)  # 高斯滤波 可以进行图像平滑处理 降噪
        equ_img = cv2.equalizeHist(blur)  # 图像增强，得到直方图均衡化后的图像
        threshold = cv2.threshold(equ_img, 127, 255, cv2.THRESH_BINARY)[1]  # 进行阈值处理，利用二值化
        if threshold is not None:
            images.append(threshold)
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
    create_csv_output("data/train_TAS.csv", POSITIVE_TRAIN_DIR, NEGATIVE_TRAIN_DIR)
