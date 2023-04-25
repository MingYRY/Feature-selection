import os
import csv
import numpy as np
import cv2
import mahotas
from mahotas import center_of_mass

ZERO_TRAIN_DIR = 'data/train_images/zero'
ONE_TRAIN_DIR = 'data/train_images/one'


def describe_shape(image):
    features = []
    blur = cv2.GaussianBlur(image, (13, 13), 0)
    threshold = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[1]

    # Closing operation
    threshold = cv2.dilate(threshold, None, iterations=4)
    threshold = cv2.erode(threshold, None, iterations=2)

    contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0]

    for c in contour:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, 1)

        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y + h, x:x + w]

        feature = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
        features.append(feature)

    return features


def get_image_feature_vector(image, positive):

    Zernike_results = describe_shape(image)

    if positive == 0:
        BIC_results = np.append(Zernike_results, 0)
        feature_set = BIC_results
    if positive == 1:
        BIC_results = np.append(Zernike_results, 1)
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
        img = cv2.imread(os.path.join(folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (13, 13), 0)  # 高斯滤波 可以进行图像平滑处理 降噪
        equ_img = cv2.equalizeHist(blur)  # 图像增强，得到直方图均衡化后的图像
        threshold = cv2.threshold(equ_img, 127, 255, cv2.THRESH_BINARY)[1]  # 进行阈值处理，利用二值化
        if threshold is not None:
            images.append(threshold)
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
    create_csv_output("data/train_Zernike.csv", ZERO_TRAIN_DIR, ONE_TRAIN_DIR)

