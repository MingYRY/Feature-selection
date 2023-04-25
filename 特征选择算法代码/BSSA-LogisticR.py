import copy
import math
import xlsxwriter
import pandas as pd
import numpy as np, numpy
import random
import sys
import time
import re
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, \
    ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class Test:
    def __init__(self, problem, alpha, beta):
        self.id = id
        self.data = problem  # FS问题
        self.a = alpha
        self.b = beta

    def nbrUn(self, solution):  # 求特征数目
        return len([i for i, n in enumerate(solution) if n == 1])

    def fitness_eq(self, X):  # 求适应度函数值，其中前半部分是分类错误率，后半部分是被选择的特征数的比率
        return (1 - self.data.evaluate(X)) * self.a + (self.nbrUn(X) / self.data.nb_attribs) * self.b

    # 计算种群适应度'''
    def CaculateFitness(self, X):
        f = []
        for i in range(X.shape[0]):  # x.shape[0]表示输出矩阵的行数，x.shape[1]表示输出矩阵的列数，这里循环100个个体次
            fitness = self.fitness_eq(X[i, :])
            f.append(fitness)
        return f  # 返回适应度函数值

    # 适应度排序'''
    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    # 根据适应度对位置进行排序'''
    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    # 边界处理'''
    def BorderCheck(self, X, ub, lb, dim):
        for j in range(dim):
            if X[j] > ub:
                X[j] = ub
            elif X[j] < lb:
                X[j] = lb
        return X

    # 二进制转换函数'''
    def transform(self, Position, dim):  # 该函数是对位置进行二进制映射
        Position_B = Position * 1
        for j in range(Position.shape[0]):  # 列数
            s = 1 / (1 + numpy.exp(-2*Position[j]))  # sigmoid 函数 对位置进行二进制映射
            if s > random.random():
                Position_B[j] = 1
            else:
                Position_B[j] = 0
        return Position_B

    # 莱维飞行
    def levy_flight(self, beta, est, alpha):
        sg = self.sigma(beta)
        u = np.random.normal(0, sg**2)
        v = abs(np.random.normal(0, 1))
        step = u/pow(v, 1/beta)
        step_size = alpha+step
        new = est+step_size  # *np.random.normal()#random.normalvariate(0,sg)
        return new

    def sigma(self, beta):
        p = math.gamma(1+beta) * math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*(pow(2,(beta-1)/2)))
        return pow(p, 1/beta)

    def sigmoid(self, x):
        try:
            return 1/(1+math.exp(-x))
        except OverflowError:
            return 0.000001

    def BSSA(self, noP, Max_iteration):
        print("The BSSA algorithm is optimizing your problem...")
        dim = self.data.nb_attribs  # 特征值的个数
        ST = 0.8  # 预警值
        PD = 0.2  # 发现者的比列，剩下的是加入者
        SD = 0.2  # 意识到有危险麻雀的比重
        PDNumber = int(noP * PD)  # 发现者数量
        SDNumber = int(noP * SD)  # 意识到有危险麻雀数量
        ub = 6
        lb = -6

        # 画图元素
        food_x = np.array(np.zeros(Max_iteration))

        Position = np.array([[0 for i in range(dim)] for j in range(noP)])  # 位置
        Position_C = np.array([[0.0 for i in range(dim)] for j in range(noP)])  # 矢量速度


        gBestScore = float("inf")
        gBest = np.array(np.zeros(dim, int))

        gWorstScore = float(0)
        gWorst = np.array(np.zeros(dim, int))

        gBest_new = np.array(np.zeros(dim, int))

        for i in range(0, noP):  # 随机粒子的位置  noP粒子群的个数
            for j in range(0, dim):
                if random.random() < 0.5:
                    Position[i][j] = 0
                else:
                    Position[i][j] = 1

        Position_former = copy.deepcopy(Position)
        for l in range(0, Max_iteration):
            fitness = self.CaculateFitness(Position)  # 计算适应度值

            fitness, sortIndex = self.SortFitness(fitness)  # 对适应度值排序

            Position = self.SortPosition(Position, sortIndex)  # 种群排序

            Position_C = self.SortPosition(Position_C, sortIndex)  # 连续位置排序

            if (gBestScore > fitness[0]):  # 更新全局最优
                gBestScore = copy.copy(fitness[0])
                gBest = copy.copy(Position[0, :])

            if (gWorstScore < fitness[-1]):  # 更新全局最优
                gWorstScore = copy.copy(fitness[-1])
                gWorst = copy.copy(Position[-1, :])

            food_x[l] = gBestScore  # 保存每次迭代的最小的适应度函数值

            # 麻雀发现者更新
            R2 = random.random()
            for i in range(PDNumber):
                for j in range(0, dim):
                    if R2 < ST:
                        Position_C[i, j] = Position_C[i, j] * numpy.exp(-(i+1) / (random.random() * Max_iteration))
                    else:
                        Position_C[i, j] = Position_C[i, j] + numpy.random.randn()
                # 越界检查
                Position_C[i] = self.BorderCheck(Position_C[i], ub, lb, dim)  # 边界检测
                # 二进制映射
                Position[i] = self.transform(Position_C[i], dim)

            # 麻雀加入者更新
            for i in range(PDNumber, noP):
                for j in range(0, dim):
                    if i > (noP / 2):
                        Position_C[i, j] = numpy.random.randn() * np.exp((gWorst[j] - Position[i, j]) / i ** 2)
                    else:
                        A = np.ones([dim, 1])
                        for a in range(dim):
                            if (random.random()) > 0.5:
                                A[a] = -1
                        AA = sum(A)/dim
#                       AA = np.dot(A, np.linalg.inv(np.dot(A.T, A)))
                        Position_C[i, j] = Position_C[0, j] + np.abs(Position[0, j] - Position[i, j]) * AA
                # 越界检查
                Position_C[i] = self.BorderCheck(Position_C[i], ub, lb, dim)  # 边界检测
                # 二进制映射
                Position[i] = self.transform(Position_C[i], dim)

            #  麻雀危险者位置更新
            Temp = range(noP)
            RandIndex = random.sample(Temp, noP)
            SDchooseIndex = RandIndex[0:SDNumber]
            for i in range(SDNumber):
                for j in range(0, dim):
                    if gBestScore < fitness[SDchooseIndex[i]]:
                        Position_C[SDchooseIndex[i], j] = gBest[j] + np.random.randn() * numpy.abs(Position[SDchooseIndex[i], j] - gBest[j])
                    elif gBestScore == fitness[SDchooseIndex[i]]:
                        K = 2 * random.random() - 1
                        Position_C[SDchooseIndex[i], j] = Position_C[SDchooseIndex[i], j] + K * \
                                                            ((np.abs(Position[SDchooseIndex[i], j] - gWorst[j])) /
                                                             (fitness[SDchooseIndex[i]] - gWorstScore + 10E-8))
                # 越界检查
                Position_C[SDchooseIndex[i]] = self.BorderCheck(Position_C[SDchooseIndex[i]], ub, lb, dim)  # 边界检测
                # 二进制映射
                Position[SDchooseIndex[i]] = self.transform(Position_C[SDchooseIndex[i]], dim)

            Position_former = copy.deepcopy(Position)

        print('gBest', gBest)
        print('gBestScore', gBestScore)  # 迭代第100次之后的最小适应度函数值
        print('fitness', food_x)
#       print(food_x)

        return self.data.evaluate(gBest), self.nbrUn(gBest), gBestScore, food_x
        # 第一项，准确度值。第二项，选择的特征数。第三项，适应度值最小的。第四项，每一次迭代的适应度函数最低值。


class FsProblem:
    def __init__(self, data):
        self.data = data
        self.nb_attribs = len(self.data.columns) - 1
        self.outPuts = self.data.iloc[:, self.nb_attribs]  # iloc 通过行号取出实际的数据
        self.classifier = DecisionTreeClassifier(min_samples_split=20, max_depth=10)  # 决策树

    def evaluate(self, solution):  # 评估分类的准确性
        list = [i for i, n in enumerate(solution) if n == 1]  # 获得选择的特征值的个数
        if (len(list) == 0):
            return 0
        df = self.data.iloc[:, list]  # 取list列表中对应特征值个数的数据
        array = df.values
        nbrAttributs = len(array[0])
        X = array[:, 0:nbrAttributs]
        Y = self.outPuts
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        results = cross_val_score(  # 求准确性
            self.classifier, X, Y, cv=cv, scoring='accuracy')

        return results.mean()


class FSData:

    def __init__(self, location, nbr_exec, particle_num, max_iter):
        self.location = location  # 数据集地址
        self.nb_exec = nbr_exec  # 运行次数
        self.dataset_name = re.search(
            '[A-Za-z\-]*.csv', self.location)[0].split('.')[0]  # 正则表达式
        self.df = pd.read_csv(self.location, header=None)  # pandas读取csv文件

        self.fsd = FsProblem(self.df)  # 调用特征选择问题
        self.particle_num = particle_num  # 蜻蜓数量
        self.max_iter = max_iter  # 最大迭代次数

        self.classifier_name = 'Decision Tree'
        # 机器学习库中的分类器
        path = 'binary_test/sheets/' + self.dataset_name + "/"
        self.instance_name = str(
            "BSSA_AB_S1-DecisionT_" + time.strftime("%m-%d-%H-%M_", time.localtime()))  # 例子的名字
        log_filename = str('binary_test/logs/' + self.dataset_name +
                           "/" + self.instance_name)

        log_file = open(log_filename + '.txt', 'w+')

    #   sys.stdout = log_file  # sys.stdout默认是映射到打开脚本的窗口

        print("[START] Dataset" + self.dataset_name + "description \n")
        print("Shape : " + str(self.df.shape) + "\n")
        print(self.df.describe())
        print("\n[END] Dataset" + self.dataset_name + "description\n")
        print("[START] Ressources specifications\n")
        print("[END] Ressources specifications\n")

        sheet_filename = str(path + self.instance_name)
        self.workbook = xlsxwriter.Workbook(sheet_filename + '.xlsx')

        self.worksheet = self.workbook.add_worksheet(
            self.classifier_name)  # 写入execl中的各个类别
        self.worksheet2 = self.workbook.add_worksheet('fitness')
        self.worksheet.write(0, 0, 'Iteration')
        self.worksheet.write(0, 1, 'Accuracy')
        self.worksheet.write(0, 2, 'N_Features')
        self.worksheet.write(0, 3, 'Fitness')
        self.worksheet.write(0, 4, 'Time')

    def attributs_to_flip(self, nb_att):

        return list(range(nb_att))

    def run(self, alpha, beta):
        t_init = time.time()  # 返回当前时间

        for itr in range(1, self.nb_exec + 1):
            print("Execution N:{0}".format(str(itr)))
            self.fsd = FsProblem(self.df)  # 将特征数据集读入到特征选择问题中
            t1 = time.time()
            test = Test(self.fsd, alpha, beta)  # 初始化
            best = test.BSSA(self.particle_num, self.max_iter)  # 粒子群数目，迭代次数
            t2 = time.time()
            print("Time elapsed for execution N:{0} : {1:.2f} s\n".format(itr, t2 - t1))
            # 将数据写在execl中
            self.worksheet.write(itr, 0, itr)
            self.worksheet.write(itr, 1, best[0])
            self.worksheet.write(itr, 2, best[1])
            self.worksheet.write(itr, 3, best[2])
            self.worksheet.write(itr, 4, t2 - t1)

            num = 0
            column = 0
            for item in best[3]:  # 输出每20次的最小的适应度函数值
                if num % 20 == 0:
                    self.worksheet2.write(column, (itr - 1), item)
                    column = column + 1
                num = num + 1

        t_end = time.time()
        print("Total execution time for dataset {0} is {1:.2f} s".format(
            self.dataset_name, t_end - t_init))
        self.workbook.close()


# Dataset
datasets = ['train_CQ', 'train_NEH', 'train_RSNA', 'train_TOM', 'train_Tumor', 'train_TWO']
data_loc_path = "binary_test/datasets/"
# 执行次数， 种群规模， 最大迭代次数
nbr_exec = 20
dragonfly_num = 20
max_iter = 201
alpha = 0.99
beta = 0.01
for dataset in datasets:
    location = data_loc_path + dataset + ".csv"
    instance = FSData(location, nbr_exec, dragonfly_num, max_iter)
    instance.run(alpha, beta)
