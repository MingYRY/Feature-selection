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
            if X[j] > ub[0, j]:
                X[j] = ub[0, j]
            elif X[j] < lb[0, j]:
                X[j] = lb[0, j]
        return X

    def jBstepBGWO(self, AD):
        Cstep = 1 / (1 + math.exp(-10 * (AD - 0.5)))
        if Cstep >= random.random():
            Bstep = 1
        else:
            Bstep = 0
        return Bstep

    def jBGWOupdate(self, X, Bstep):
        if (X + Bstep) >= 1:
            Y = 1
        else:
            Y = 0
        return Y

    def sigma(self, beta):
        p = math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                    math.gamma((1 + beta) / 2) * beta * (pow(2, (beta - 1) / 2)))
        return pow(p, 1 / beta)

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.000001

    def BGWO(self, pop, Max_iteration):
        print("The BGWO algorithm is optimizing your problem...")
        # Parameters
        dim = self.data.nb_attribs  # 特征值的个数
        ub = 6
        lb = -6

        # Dimension
        if np.size(lb) == 1:
            ub = ub * np.ones([1, dim], dtype='float')
            lb = lb * np.ones([1, dim], dtype='float')

        # Initialize position
        X = np.array([[0.0 for i in range(dim)] for j in range(pop)])  # 位置

        food_x = np.array(np.zeros(Max_iteration))

        gBestScore = float("inf")
        gBest = np.array(np.zeros(dim, int))

        # Fitness at first iteration
        Xalpha = np.zeros([1, dim], dtype='float')
        Xbeta = np.zeros([1, dim], dtype='float')
        Xdelta = np.zeros([1, dim], dtype='float')

        Falpha = float('inf')
        Fbeta = float('inf')
        Fdelta = float('inf')

        for i in range(0, pop):  # 随机粒子的位置  noP粒子群的个数
            for j in range(0, dim):
                if random.random() < 0.5:
                    X[i][j] = 0
                else:
                    X[i][j] = 1

        fitness = self.CaculateFitness(X)  # 计算适应度值
        for i in range(pop):
            if fitness[i] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fitness[i]

            if fitness[i] < Fbeta and fitness[i] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fitness[i]

            if fitness[i] < Fdelta and fitness[i] > Fbeta and fitness[i] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fitness[i]

        # Pre
        curve = np.zeros([1, Max_iteration], dtype='float')

        t = 0

        # curve[0, t] = Falpha.copy()
        food_x[t] = Falpha  # 保存每次迭代的最小的适应度函数值
        t += 1

        while t < Max_iteration:
            # Coefficient decreases linearly from 2 to 0
            a = 2 - t * (2 / Max_iteration)

            for i in range(pop):
                for d in range(dim):
                    # Parameter C (3.4)
                    C1 = 2 * random.random()
                    C2 = 2 * random.random()
                    C3 = 2 * random.random()
                    # Compute Dalpha, Dbeta & Ddelta (3.5)
                    Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                    Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                    Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                    # Parameter A (3.3)
                    A1 = 2 * a * random.random() - a
                    # Compute X1, X2 & X3 (3.6)
                    Bstep1 = self.jBstepBGWO(A1 * Dalpha)
                    Bstep2 = self.jBstepBGWO(A1 * Dbeta)
                    Bstep3 = self.jBstepBGWO(A1 * Ddelta)

                    X1 = self.jBGWOupdate(Xalpha[0, d], Bstep1)
                    X2 = self.jBGWOupdate(Xbeta[0, d], Bstep2)
                    X3 = self.jBGWOupdate(Xdelta[0, d], Bstep3)

                    # Update wolf (3.7)
                    r = random.random()
                    if r < 1 / 3:
                        X[i, d] = X1
                    elif r < 2 / 3 and r >= 1 / 3:
                        X[i, d] = X2
                    else:
                        X[i, d] = X3

            fitness = self.CaculateFitness(X)  # 计算适应度值
            # Fitness
            for i in range(pop):
                if fitness[i] < Falpha:
                    Xalpha[0, :] = X[i, :]
                    Falpha = fitness[i]

                if fitness[i] < Fbeta and fitness[i] > Falpha:
                    Xbeta[0, :] = X[i, :]
                    Fbeta = fitness[i]

                if fitness[i] < Fdelta and fitness[i] > Fbeta and fitness[i] > Falpha:
                    Xdelta[0, :] = X[i, :]
                    Fdelta = fitness[i]

            food_x[t] = Falpha  # 保存每次迭代的最小的适应度函数值
            # curve[0, t] = Falpha.copy()
            t += 1

        gBestScore = copy.copy(Falpha)
        gBest = copy.copy(Xalpha[0, :])
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
        self.classifier = DecisionTreeClassifier(min_samples_split=20, max_depth=10)  # 决策树分类

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
            "BGWO-DecisionT_" + time.strftime("%m-%d-%H-%M_", time.localtime()))  # 例子的名字
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
            best = test.BGWO(self.particle_num, self.max_iter)  # 粒子群数目，迭代次数
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
datasets = ['train_TWO']
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
