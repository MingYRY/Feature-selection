import numpy as np
import cv2
import os
import csv

ZERO_TRAIN_DIR = 'data/train_images/zero'
ONE_TRAIN_DIR = 'data/train_images/one'

EPS = 0.00000000000000001

# 建立Gabor滤波器
def build_filters():
    filters = []

    # Size of Gabor kernel
    ksize = 31

    # For different orientations
    for theta in np.arange(0, np.pi, np.pi / 4):
        # And different wavelengths of the sinusoidal factor
        for lamb in np.arange(np.pi / 4, np.pi, np.pi / 4):
            # Get a filter
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamb, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)

    return filters

# Given an image and a set of filters, derive the response matrices
def process(img, filters):

    responses = []
    # accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        responses.append(fimg)
        # np.maximum(accum, fimg, accum)

    return responses


# Given a response matrix, compute for the local energy
# Local Energy = summing up the squared value of each matrix value from a response matrix
def get_local_energy(matrix):
    local_energy = 0.0
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            val = int(matrix[row][col]) * int(matrix[row][col])
            local_energy = local_energy + val

    # Divide by the highest possible value, which is 255^2 * (100 x 100)
    # to normalize values from 0 to 1, and replace 0s with EPS value to work with NB
    local_energy = local_energy / 650250000
    return EPS if local_energy == 0 else local_energy


# Given a response matrix, compute for the mean amplitude
# Mean Amplitude = sum of absolute values of each matrix value from a response matrix
def get_mean_amplitude(matrix):
    mean_amp = 0.0

    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            val = abs(int(matrix[row][col]))
            mean_amp = mean_amp + val

    # Divide by the highest possible value, which is 255 * (100 x 100)
    # to normalize values from 0 to 1, and replace 0s with EPS value to work with NB
    mean_amp = mean_amp / 2550000
    return EPS if mean_amp == 0 else mean_amp

# 返回给定目录中的图像列表
# Return a list of images from the given directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        blur = cv2.GaussianBlur(gray, (13, 13), 0)  # 高斯滤波 可以进行图像平滑处理 降噪
        equ_img = cv2.equalizeHist(blur)  # 图像增强，得到直方图均衡化后的图像
        threshold = cv2.threshold(equ_img, 127, 255, cv2.THRESH_BINARY)[1]  # 进行阈值处理，利用二值化
        if threshold is not None:
            images.append(threshold)
    return images


# Get the feature vector (local energy/mean amplitude from response matrices) of an image
# This function is called when bulding the CSV files or when processing each frame
def get_image_feature_vector(image, filters, positive):
    response_matrices = process(image, filters)

    local_energy_results = []
    mean_amplitude_results = []

    for matrix in response_matrices:
        local_energy = get_local_energy(matrix)
        mean_amplitude = get_mean_amplitude(matrix)
        local_energy_results.append(local_energy)
        mean_amplitude_results.append(mean_amplitude)

    if positive == 0:
        feature_set = local_energy_results + mean_amplitude_results + [0]
    if positive == 1:
        feature_set = local_energy_results + mean_amplitude_results + [1]

    return feature_set


# Get all feature vectors of a set of images, used when building the CSV files
def get_all_image_feature_vectors(images, positive):
    filters = build_filters()
    feature_sets = []

    for image in images:
        feature_set = get_image_feature_vector(image, filters, positive)
        feature_sets.append(feature_set)

    return feature_sets


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
    create_csv_output("data/train_Gabor.csv", ZERO_TRAIN_DIR, ONE_TRAIN_DIR)
