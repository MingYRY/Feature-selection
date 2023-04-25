import cv2
import numpy as np
import os
import mahotas as mt
import csv

POSITIVE_TRAIN_DIR = 'data/train_images/positives'
NEGATIVE_TRAIN_DIR = 'data/train_images/negatives'
from numpy import histogram


# 灰度特征
def GrayFea(faultness):
    '''
    :param faultness: 灰度图像
    :return: list->(4)
    '''
    hist0 = cv2.calcHist([faultness], [0], None, [256], [0, 255])
    h, w = faultness.shape
    hist = hist0 / (h * w)

    # 灰度平均值
    mean_gray = 0
    for i in range(len(hist)):
        mean_gray += i * hist[i]

    # 灰度方差
    var_gray = 0
    for i in range(len(hist)):
        var_gray += hist[i] * (i - mean_gray) ** 2

    # 能量
    ##归一化
    max_ = np.max(hist)
    min_ = np.min(hist)
    hist_ = (hist - min_) / (max_ - min_)
    ##求解能量
    energy = 0
    for i in range(len(hist_)):
        energy += hist_[i] ** 2

    # 灰度对比度
    con = np.max(faultness) - np.min(faultness)
    gray_fea = [mean_gray[0], var_gray[0], energy[0], con]
    return gray_fea


def get_image_feature_vector(image, positive=None):
    Gary_results = GrayFea(image)

    if positive is None:
        feature_set = Gary_results
    else:
        pos_num = 1 if positive else 0
        Haralick_results = np.append(Gary_results, pos_num)
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
        if gray is not None:
            images.append(gray)
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
    create_csv_output("data/train_Gray.csv", POSITIVE_TRAIN_DIR, NEGATIVE_TRAIN_DIR)
