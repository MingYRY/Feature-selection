import cv2
import numpy as np
import math
import os
import mahotas as mt
import csv

EPS = 0.00000000000000001
ZERO_TRAIN_DIR = 'data/train_images/zero'
ONE_TRAIN_DIR = 'data/train_images/one'


def coarseness(image, kmax):
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax, w, h])
    horizon = np.zeros([kmax, w, h])
    vertical = np.zeros([kmax, w, h])
    Sbest = np.zeros([w, h])

    for k in range(kmax):
        window = np.power(2, k)
        for wi in range(w)[window:(w - window)]:
            for hi in range(h)[window:(h - window)]:
                average_gray[k][wi][hi] = np.sum(image[wi - window:wi + window, hi - window:hi + window])
        for wi in range(w)[window:(w - window - 1)]:
            for hi in range(h)[window:(h - window - 1)]:
                horizon[k][wi][hi] = average_gray[k][wi + window][hi] - average_gray[k][wi - window][hi]
                vertical[k][wi][hi] = average_gray[k][wi][hi + window] - average_gray[k][wi][hi - window]
        horizon[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))
        vertical[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:, wi, hi])
            h_max_index = np.argmax(horizon[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            v_max_index = np.argmax(vertical[:, wi, hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2, index)

    fcrs = np.mean(Sbest)
    return fcrs


def contrast(image):
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4))
    v = np.var(image)
    std = np.power(v, 0.5)
    alfa4 = m4 / np.power(v, 2)
    fcon = std / np.power(alfa4, 0.25)
    return fcon


def directionality(image):
    image = np.array(image, dtype='int64')
    h = image.shape[0]
    w = image.shape[1]
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    deltaH = np.zeros([h, w])
    deltaV = np.zeros([h, w])
    theta = np.zeros([h, w])

    # calc for deltaH
    for hi in range(h)[1:h - 1]:
        for wi in range(w)[1:w - 1]:
            deltaH[hi][wi] = np.sum(np.multiply(image[hi - 1:hi + 2, wi - 1:wi + 2], convH))
    for wi in range(w)[1:w - 1]:
        deltaH[0][wi] = image[0][wi + 1] - image[0][wi]
        deltaH[h - 1][wi] = image[h - 1][wi + 1] - image[h - 1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w - 1] = image[hi][w - 1] - image[hi][w - 2]

    # calc for deltaV
    for hi in range(h)[1:h - 1]:
        for wi in range(w)[1:w - 1]:
            deltaV[hi][wi] = np.sum(np.multiply(image[hi - 1:hi + 2, wi - 1:wi + 2], convV))
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h - 1][wi] = image[h - 1][wi] - image[h - 2][wi]
    for hi in range(h)[1:h - 1]:
        deltaV[hi][0] = image[hi + 1][0] - image[hi][0]
        deltaV[hi][w - 1] = image[hi + 1][w - 1] - image[hi][w - 1]

    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
    deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

    # calc the theta
    for hi in range(h):
        for wi in range(w):
            if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
                theta[hi][wi] = 0
            elif (deltaH[hi][wi] == 0):
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

    n = 16
    t = 12
    cnt = 0
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if ((deltaG_vec[k] >= t) and (theta_vec[k] >= (2 * ni - 1) * np.pi / (2 * n)) and (
                    theta_vec[k] < (2 * ni + 1) * np.pi / (2 * n))):
                hd[ni] += 1
    hd = hd / np.mean(hd)
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]
    return fdir, theta


def linelikeness(image, theta, dist):
    image = np.array(image, dtype='int64')
    n = 16
    h = image.shape[0]
    w = image.shape[1]
    pi = 3.1415926
    dcm = np.zeros((8, n, n))
    # 8 direction
    dir = np.float32([[[1, 0, -dist], [0, 1, -dist]],
                      [[1, 0, -dist], [0, 1, 0]],
                      [[1, 0, dist], [0, 1, 0]],
                      [[1, 0, 0], [0, 1, -dist]],
                      [[1, 0, 0], [0, 1, dist]],
                      [[1, 0, dist], [0, 1, -dist]],
                      [[1, 0, dist], [0, 1, 0]],
                      [[1, 0, dist], [0, 1, dist]]
                      ])
    cooccurrence_matrixes = []
    for i in range(8):
        # move matrix_theta along direction above
        cooccurrence_matrixes.append(cv2.warpAffine(theta, dir[i], (w, h)))

    for m1 in range(1, n):
        for m2 in range(1, n):
            for d in range(8):
                # judgement, if 2 pixel(original, moved pic) in the same range, return true
                # sorry for split judge function, because I don't know how to write them together(I'm new in python)
                m_theta_bottom = (theta >= ((2 * (m1 - 1) * pi) / (2 * n)))
                m_theta_top = (theta < (((2 * (m1 - 1) + 1) * pi) / (2 * n)))
                m_theta = np.logical_and(m_theta_bottom, m_theta_top)

                m_ccoccurrence_matrixes_bottom = (cooccurrence_matrixes[d] >= ((2 * (m2 - 1) * pi) / (2 * n)))
                m_ccoccurrence_matrixes_top = (cooccurrence_matrixes[d] < (((2 * (m2 - 1) + 1) * pi) / (2 * n)))
                m_ccoccurrence_matrixes = np.logical_and(m_ccoccurrence_matrixes_bottom, m_ccoccurrence_matrixes_top)

                dcm_matrix = np.logical_and(m_theta, m_ccoccurrence_matrixes)
                dcm_matrix = dcm_matrix.astype(int)
                dcm[d][m1][m2] = np.sum(dcm_matrix)
    matrix_f = np.zeros((1, 8))
    matrix_g = np.zeros((1, 8))

    # calculate the angle of 8 direction, and sum them up
    for i in range(n):
        for j in range(n):
            for d in range(8):
                matrix_f[0][d] += dcm[d][i][j] * (math.cos((i - j) * 2 * pi / n))
                matrix_g[0][d] += dcm[d][i][j]
                # set in range (0,1)
    matrix_res = matrix_f / (matrix_g + EPS)
    # return the max one ,can describe how this picture texture's "direction" move
    res = np.max(matrix_res)
    return res


def regularity(image, filter):
    image = np.array(image, dtype='int64')
    h = image.shape[0]
    w = image.shape[1]
    k = 0
    crs = []
    con = []
    dire = []
    lin = []
    for i in range(1, h, filter):
        for j in range(1, w, filter):
            con.append(contrast(image[i:i + filter - 1, j: j + filter - 1]))
            [s, sita] = directionality(image[i:i + filter - 1, j: j + filter - 1])
            dire.append(s)
            lin.append(linelikeness(image[i:i + filter - 1, j: j + filter - 1], sita, 4)*10)
            crs.append(coarseness(image[i:i + filter - 1, j:j + filter - 1], 5))
    Dcrs = np.std(crs, ddof=1)
    Dcon = np.std(con, ddof=1)
    Ddir = np.std(dire, ddof=1)
    Dlin = np.std(lin, ddof=1)
    Freg = 1 - (Dcrs + Dcon + Ddir + Dlin) / 4 / 100
    return Freg


def roughness(fcrs, fcon):
    return fcrs + fcon


def get_image_feature_vector(image, positive=None):
    Tamura_results = []

    fcrs = coarseness(image, 5)
    fcon = contrast(image)
    fdir = directionality(image)[0]
    flin = linelikeness(image, fdir, 4)
    freg = regularity(image, 64)
    frgh = roughness(fcrs, fcon)

    Tamura_results.append(fcrs)
    Tamura_results.append(fcon)
    Tamura_results.append(fdir)
    Tamura_results.append(flin)
    Tamura_results.append(freg)
    Tamura_results.append(frgh)

    if positive == 0:
        BIC_results = np.append(Tamura_results, 0)
        feature_set = BIC_results
    if positive == 1:
        BIC_results = np.append(Tamura_results, 1)
        feature_set = BIC_results


    return feature_set


# Get all feature vectors of a set of images, used when building the CSV files
def get_all_image_feature_vectors(images, positive):
    feature_sets = []
    A = 0
    for image in images:
        A = A + 1
        print("图片：", A)
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
        gray = cv2.resize(gray, (100, 100))
        # blur = cv2.GaussianBlur(gray, (13, 13), 0)  # 高斯滤波 可以进行图像平滑处理 降噪
        # equ_img = cv2.equalizeHist(blur)  # 图像增强，得到直方图均衡化后的图像
        # threshold = cv2.threshold(equ_img, 127, 255, cv2.THRESH_BINARY)[1]  # 进行阈值处理，利用二值化
        if gray is not None:
            images.append(gray)
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
    create_csv_output("data/train_Tamura.csv", ZERO_TRAIN_DIR, ONE_TRAIN_DIR)
