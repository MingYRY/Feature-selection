import cv2
import numpy as np
import os
import csv

ZERO_TRAIN_DIR = 'data/train_images/zero'
ONE_TRAIN_DIR = 'data/train_images/one'
np.set_printoptions(suppress=True)  # 定义输出的精度


def sys_moments(img):
    '''
    opencv_python自带求矩以及不变矩的函数
    :param img: 灰度图像，对于二值图像来说就只有两个灰度0和255
    :return: 返回以10为底对数化后的hu不变矩
    '''
    moments = cv2.moments(img)  # 返回的是一个字典，三阶及以下的几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)
    humoments = cv2.HuMoments(moments)  # 根据几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)计算出hu不变矩
    # 因为直接计算出来的矩可能很小或者很大，因此取对数好比较,这里的对数底数为e,通过对数除法的性质将其转换为以10为底的对数，一般是负值，因此加一个负号将其变为正的
    humoment = -(np.log(np.abs(humoments))) / np.log(10)
    return humoment


def get_image_feature_vector(image, positive):

    BIC_results = sys_moments(image)

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
    create_csv_output("data/train_Hu.csv", ZERO_TRAIN_DIR, ONE_TRAIN_DIR)