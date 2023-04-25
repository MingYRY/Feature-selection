# -*- coding: utf-8 -*-
"""
Created on Thirsday March 21  2019

@author:
% _____________________________________________________
% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems,
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% _____________________________________________________

"""
import random
import math
import matplotlib.pyplot as plt  # 用来画图的包
import numpy
import numpy as np
import pandas as pd  # pandas用来做数据处理的包
from sklearn.model_selection import train_test_split  # 将数据集划分为训练集和测试集   train_test_split表示训练测试和划分
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC  # 支持向量机  ？？？？
from sklearn.preprocessing import StandardScaler  # 数据预处理之数据标准化
from sklearn.svm import SVC  # 支持向量机
from sklearn.metrics import accuracy_score  # accuracy_score是准确率
import time
import xlsxwriter  # 进行execl操作  xlsxwriter总之比xlwt好


# 获取所有样本的三个目标函数的值
def get_ObjectFunction(population, dim, x, y, z):
    f = []
    for i in range(population.shape[0]):  # x.shape[0]表示输出矩阵的行数，x.shape[1]表示输出矩阵的列数，这里循环100个个体次
        while (sum(population[i]) < 2):  # （选择的特征最少得有一个）对于全是false 则更新个体
            population[i] = numpy.random.randint(0, 2, (1, dim))
        column_use = []
        column_use = (population[i] == 1)  # column_use得到的是一个只含有False、True的列表
        x_test = x.columns[column_use]  # 得到值为true的所有列号，这些列号形成一个列表
        z_test = z.columns[column_use]  # 得到成本为ture的特征值列号
        clf = OneVsOneClassifier(SVC(C=1.0, random_state=20, kernel='rbf', cache_size=1500))  # 实例化对象clf
        # x[x_test]表示选中所有值为1的所有列，形成一个矩阵，作为新的数据集
        X_train, X_test, y_train, y_test = train_test_split(x[x_test], y, test_size=0.7, random_state=7)
        clf.fit(X_train, y_train)  # 填充训练数据进行训练--->相当于训练模型
        fitness_1 = 1 - accuracy_score(y_test, clf.predict(X_test))  # 正确率
        costSum = 0
        for i in z_test:
            costSum+=z[i]
        number = [fitness_1, 1.0 * (sum(column_use))/dim, costSum[0]]
        f.append(number)
    return f  # 返回三个函数值列表


def single_get_ObjectFunction(population, dim, x, y, z):
    while (sum(population) < 2):  # （选择的特征最少得有一个）对于全是false 则更新个体
        population = numpy.random.randint(0, 2, (1, dim))
    column_use = (population == 1)  # column_use得到的是一个只含有False、True的列表
    x_test = x.columns[column_use]  # 得到值为true的所有列号，这些列号形成一个列表
    z_test = z.columns[column_use]  # 得到成本为ture的特征值列号
    clf = OneVsOneClassifier(SVC(C=1.0, random_state=20, kernel='rbf', cache_size=1500))  # 实例化对象clf
    # x[x_test]表示选中所有值为1的所有列，形成一个矩阵，作为新的数据集
    X_train, X_test, y_train, y_test = train_test_split(x[x_test], y, test_size=0.7, random_state=7)
    clf.fit(X_train, y_train)  # 填充训练数据进行训练--->相当于训练模型
    fitness_1 = 1 - accuracy_score(y_test, clf.predict(X_test))  # 正确率
    costSum = 0
    for i in z_test:
        costSum += z[i]
    number = [fitness_1, 1.0 * (sum(column_use)) / dim, costSum[0]]
    return number


def dominates(x, y):
    if all(x <= y) and any(x < y):
        return True  # x支配y
    else:
        return False


def updateArchive(Archive_X, Archive_F, population, particles_F):  # 怎么求最里边的那几个数据
    Archive_temp_X = numpy.vstack((Archive_X, population))  # 纵向合并 136，36
    Archive_temp_F = numpy.vstack((Archive_F, particles_F))  # 纵向合并 136，3
    o = numpy.zeros(Archive_temp_F.shape[0])  # 表示矩阵的行数Archive_temp_X.shape[0] 136
    for i in range(0, Archive_temp_F.shape[0]):
        for j in range(0, Archive_temp_F.shape[0]):
            if i != j:
                if dominates(Archive_temp_F[j], Archive_temp_F[i]):  # 每一个档案中的个体与其他所有个体比较，若被支配则标记为1
                    o[i] = 1
                    break
        pass
    Archive_member_no = 0
    Archive_X_updated = []
    Archive_F_updated = []
    for i in range(Archive_temp_F.shape[0]):  # 将非支配解放入到档案集中
        if o[i] == 0:
            Archive_member_no = Archive_member_no + 1
            Archive_X_updated.append(Archive_temp_X[i])
            Archive_F_updated.append(Archive_temp_F[i])
    return Archive_X_updated, Archive_F_updated, Archive_member_no


def RankingProcess(Archive_F, obj_no):  # 排序流程 是对搜索空间划分网格，获得当前档案集的位置
    # 传递过来的是个数组类型
    if len(Archive_F) == 1:  # 如果数组的长度是1
        my_min = [Archive_F[0][0], Archive_F[0][1], Archive_F[0][2]]
        my_max = [Archive_F[0][0], Archive_F[0][1], Archive_F[0][2]]
    else:
        my_min = [min(Archive_F[:, 0]), min(Archive_F[:, 1]), min(Archive_F[:, 2])]  # my_min表示输出每一列的最小值
        my_max = [max(Archive_F[:, 0]), max(Archive_F[:, 1]), max(Archive_F[:, 2])]  # my_max表示输出每一列的最大值

    r = [(my_max[0] - my_min[0]) / 10, (my_max[1] - my_min[1]) / 10, (my_max[2] - my_min[2]) / 10]
    # 最大值-最小值得到一个片段，将纵轴和横轴分成多少份
    ranks = numpy.zeros(len(Archive_F))
    for i in range(len(Archive_F)):  # 判断每个解周围有几个解
        ranks[i] = 0
        for j in range(len(Archive_F)):
            flag = 0  # 一个标志，以查看该点是否在所有维度上的附近
            for k in range(obj_no):
                if math.fabs(Archive_F[j][k] - Archive_F[i][k]) <= r[k]:
                    flag = flag + 1
            if flag == obj_no:
                ranks[i] = ranks[i] + 1
                pass
            pass
        pass
    return ranks  # 判断每个帕累托解周围有几个解


def RouletteWheelSelection(weights):  # eg [1. 2. 1. 2. 4.]
    accumulation = numpy.cumsum(weights)  # accumulation=[1. 3. 4. 6. 10.]
    p = random.random()
    end = accumulation[accumulation.shape[0] - 1]  # accumulation.shape[0]=5，最后一个数，也相当于所有数字的和
    list = []  # list最后得[0, 1/10, 3/10, 4/10, 6/10, 1]
    list.append(0)
    for i in range(weights.shape[0]):
        list.append(1.0 * accumulation[i] / end)  #
    chosen_index = -1
    for i in range(accumulation.shape[0]):
        if (p < list[i + 1] and p > list[i]):
            chosen_index = i
            break
    pass
    o = chosen_index
    return o


def handleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize):  # 处理存档值已经满的情况
    for i in range(len(Archive_F) - ArchiveMaxSize):
        index = RouletteWheelSelection(Archive_mem_ranks)
        Archive_X = numpy.vstack(
            (Archive_X[0:index], Archive_X[index + 1:Archive_member_no]))  # vstack表示纵向合并  a[0:2],截取矩阵a的第一行到第二行，前闭后开
        Archive_F = numpy.vstack((Archive_F[0:index], Archive_F[index + 1:Archive_member_no]))  # hstack表示横向合并
        Archive_mem_ranks = numpy.hstack((Archive_mem_ranks[0:index], Archive_mem_ranks[index + 1:Archive_member_no]))
        Archive_member_no = Archive_member_no - 1

    Archive_X_Chopped = 1.0 * Archive_X
    Archive_F_Chopped = 1.0 * Archive_F
    Archive_mem_ranks_updated = 1.0 * Archive_mem_ranks
    return Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no
    pass


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step


def transform(population1, chromosome_length):   # 该函数是对位置进行二进制映射
    population_fit = population1 * 1
    p = random.random()
    for j in range(population1.shape[0]):  # 列数
        a = 1 / (1 + numpy.exp(-population1[j]))  # sigmoid 函数 对位置进行二进制映射
        if a > 0.5:
            population_fit[j] = 1
        if a < 0.5:
            population_fit[j] = 0
    while (sum(population_fit) < 2):  # 有全False时重新生成个体-
        b = 1-2*numpy.random.rand(chromosome_length)
        population_fit = b * 1
        for j in range(population_fit.shape[0]):
            a = 1 / (1 + numpy.exp(-population_fit[j]))
            if a > 0.5:
                population_fit[j] = 1
            if a < 0.5:
                population_fit[j] = 0
    return (population_fit)


def delete(list):  # 改变种群中的相同个体，目的应该是增加多样性
    for i in range(list.shape[0]):
        for j in range(list.shape[0]):
            if (all(list[i] == list[j])and i!=j):

                print('相同的', i, j, all(list[i] == list[j]))

                list[j]= numpy.random.randint(0, 2, (1, dim))
                print('nest', j, '改为', list[j])
    return(list)


def fast_non_dominated_sort(population, values1, values2, values3, particles_F):
    S = [[] for i in range(0, len(values1))]  # 空的列表
    # SP: 被支配个体集合，该量是可行解空间中所有被个体p支配的个体组成的集合
    # NP：该量是在可行解空间中可以支配个体p的所有个体的数目。
    # 种群中所有个体的sp进行初始化 这里的len(value1)=pop_size 种群个体数
    front = [[]]
    # 分层集合,二维列表中包含第n个层中,有那些个体
    n = [0 for i in range(0, len(values1))]  # 生成两个0的列表
    rank = [0 for i in range(0, len(values1))]
    # 评级
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        # 寻找第p个个体和其他个体的支配关系
        # 将第p个个体的sp和np初始化
        for q in range(0, len(values1)):
            # step2:p > q 即如果p支配q,则
            if (values1[p] <= values1[q] and values2[p] <= values2[q] and values3[p] < values3[q]) or \
                    (values1[p] <= values1[q] and values2[p] < values2[q] and values3[p] <= values3[q]) or\
                    (values1[p] < values1[q] and values2[p] <= values2[q] and values3[p] <= values3[q]):
                # 支配判定条件:当且仅当,对于任取i属于{1,2},都有fi(p)>fi(q),符合支配.或者当且仅当对于任意i属于{1,2},有fi(p)>=fi(q),且至少存在一个j使得fj(p)>f(q)  符合弱支配
                if q not in S[p]:
                    # 同时如果q不属于sp将其添加到sp中
                    S[p].append(q)
            # 如果q支配p
            elif (values1[q] <= values1[p] and values2[q] <= values2[p] and values3[q] < values3[p]) or \
                    (values1[q] <= values1[p] and values2[q] < values2[p] and values3[q] <= values3[p]) or \
                    (values1[q] < values1[p] and values2[q] <= values2[p] and values3[q] <= values3[p]):
                # 则将np+1
                n[p] = n[p] + 1
        if n[p] == 0:
            # 找出种群中np=0的个体
            rank[p] = 0
            # 将其从pt中移去
            if p not in front[0]:
                # 如果p不在第0层中
                # 将其追加到第0层中
                front[0].append(p)

    i = 0
    while (front[i] != []):
        # 如果分层集合为不为空，
        Q = []
        for p in front[i]:
            for q in S[p]:  # q为SP中的个体的序号
                n[q] = n[q] - 1
                # 则将fk中所有给对应的个体np-1
                if (n[q] == 0):
                    # 如果nq==0
                    rank[q] = i + 1

                    if q not in Q:
                        Q.append(q)
        i = i + 1
        # 并且k+1
        front.append(Q)

    del front[len(front) - 1]
    print('sort_front[0]', front[0])
    population_sort = []
    population_sort_fit = []
    # 对种群根据帕累托集进行非支配排序
    for n in range(len(front)):
        for m in range(len(front[n])):
            population_sort.append(population[front[n][m]])
            population_sort_fit.append(particles_F[front[n][m]])
    population_sort = numpy.array(population_sort)
    population_sort_fit = numpy.array(population_sort_fit)
    return (population_sort, population_sort_fit)
# 下面是主程序

# 在此处更换数据集的名字👇

inputname = 'hypothyroid'  # Zoo 16个特征   Krvskp 36个特征

# 只改上面就可以的黑盒     👆别处不要动
inputdata = 'C:/Users/dell/Desktop/data-cost/' + inputname + ".csv"  # inputdata是数据集的存放地址
inputdata1 = 'C:/Users/dell/Desktop/data-cost/' + inputname + '-cost' + ".csv"

dataset = pd.read_csv(inputdata, header=None)
dataset1 = pd.read_csv(inputdata1, header=None)

workbook = xlsxwriter.Workbook('C:/Users/dell/Desktop/data-cost/' + inputname + '_MOSSA9' + '.xlsx')
worksheet1 = workbook.add_worksheet("30代")
worksheet2 = workbook.add_worksheet("60代")
worksheet3 = workbook.add_worksheet("100代")
ColName = ["f1（错误率）", 'f2（特征子集/总特征）', 'f3(最小成本)']
for i in range(len(ColName)):
    worksheet1.write(0, i, ColName[i])
    worksheet2.write(0, i, ColName[i])
    worksheet3.write(0, i, ColName[i])
x = dataset.iloc[:, 0:-1]  # 特征——>去掉了最后一列标签，x表示所有的特征
# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
# fit_transform()先拟合数据，再标准化
x = pd.DataFrame(StandardScaler().fit_transform(x))  # 利用支持向量机最好进行数据归一化预处理
y = dataset.iloc[:, -1]  # 标签——>只取最后一列，y表示标签列
z = dataset1.iloc[:]  # 成本

population_size = 30  # 每次生成30条数据，搜索代理数
T = 100  # 迭代次数
number = []  # 特征数
dim = x.shape[1]  # 标签列数，剩下有多少特征列，Krvskp有36个特征列
obj_no = 3  # 目标数是3,即有三个目标函数f1=1-E,f2=C/R，f3=sum(成本)
lb = -6  # 搜索空间的上下界
ub = 6  # 搜索空间的上下界

Archive_F1 = []
Archive_F2 = []

ArchiveMaxSize = 100  # 存档最大值
Archive_member_no = 0  # 存档数
Archive_X = numpy.zeros((ArchiveMaxSize, dim))  # 生成一个100行36列的值为0的矩阵  用来存储样本的值，即一条x的值
Archive_F = numpy.ones((ArchiveMaxSize, obj_no)) * float("inf")  # Archive_F表示100行3列的值为无穷大的矩阵  用来存储三个目标函数的值

Gbest_Location = numpy.zeros(dim)
Gbest_fitness = float("inf") * numpy.ones(3)  # change this to -inf for maximization problems

ST = 0.8  # 预警值
PD = 0.2  # 发现者的比列，剩下的是加入者
SD = 0.2  # 意识到有危险麻雀的比重
PDNumber = int(population_size * PD)  # 发现者数量  30*0.2=6
SDNumber = int(population_size * SD)  # 意识到有危险麻雀数量 30*0.2=6

# Initialize the locations of Harris' hawks
population = numpy.random.randint(0, 2, (population_size, dim))  #

t = 0
totle_time = 0
while t < T:
    start = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    population = delete(population)
    # fitness of locations
    particles_F = get_ObjectFunction(population, dim, x, y, z)  # 应该有36行，3列
    particles_F = numpy.array(particles_F)

    # Update the location of Gbest
    Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F,population, particles_F)
    if Archive_member_no > ArchiveMaxSize:  # 进行档案集的越界判断
        Archive_mem_ranks = RankingProcess(numpy.array(Archive_F), obj_no)  # 对每个解周围的解进行筛选
        Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = \
            handleFullArchive(numpy.array(Archive_X), numpy.array(Archive_F),
                              Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
        pass
    else:
        Archive_mem_ranks = RankingProcess(numpy.array(Archive_F), obj_no)  # 包含每个个体对应的周围解的个数

    # 选择最优个体
    index = RouletteWheelSelection(1 / numpy.array(Archive_mem_ranks))  # 此处是轮盘选择
    print('index(Gbest):', index)
    if index == -1:
        index = 0
    Gbest_fitness = Archive_F[index]
    Gbest_Location = Archive_X[index]

    # 对种群进行排序
    particles_F_rev = particles_F.T  # 对函数
    population, particles_F = fast_non_dominated_sort(population, particles_F_rev[0],
                                                      particles_F_rev[1], particles_F_rev[2], particles_F)
    # 选择最差个体 这是轮盘选择，还可以选择排序之后的最差的
    # 选择的是整个过程中的最差的个体
    index = RouletteWheelSelection(Archive_mem_ranks)  # 此处是轮盘选择
    print('index(Gworst):', index)
    if index == -1:
        index = 0
    Gworst_fitness = Archive_F[index]
    Gworst_Location = Archive_X[index]

    if t == 30:
        print("以上完成了{}次迭代".format(t))
        for q in range(len(Archive_F)):  # 30代时，档案集中的最小特征值
            Archive_F1.append(Archive_F[q])
    if t == 60:
        print("以上完成了{}次迭代".format(t))
        for q in range(len(Archive_F)):  # 60代时，档案集中的最小特征值
            Archive_F2.append(Archive_F[q])

    # 麻雀发现者更新
    R2 = random.random()
    for i in range(PDNumber):
        if R2 < ST:
            population[i, :] = population[i, :] * numpy.exp(-i/(random.random()*T))
        else:
            population[i, :] = population[i, :] + numpy.random.randn()*numpy.ones([1, dim])
        # 越界检查
        for j in range(population.shape[1]):
            if population[i][j] > 6:  # 速度向量越界检查
                population[i][j] = 6
            if population[i][j] < -6:
                population[i][j] = -6
        # 二进制映射
        population[i] = transform(population[i], dim)

    # 麻雀加入者更新
    for i in range(PDNumber+1, population_size):
        if i > (population_size / 2):
            population[i, :] = numpy.random.randn()*np.exp((Gworst_Location-population[i, :])/i**2)
        else:
            A = np.ones([dim, 1])
            for a in range(dim):
                if (random.random()) > 0.5:
                    A[a] = -1
            AA = np.dot(A, np.linalg.inv(np.dot(A.T, A)))
            population[i, :] = population[0, :] + np.abs(Gbest_Location - population[i, :]) * AA.T * \
                                                                                        numpy.ones([1, dim])
        # 越界检查
        for j in range(population.shape[1]):
            if population[i][j] > 6:  # 速度向量越界检查
                population[i][j] = 6
            if population[i][j] < -6:
                population[i][j] = -6
        # 二进制映射
        population[i] = transform(population[i], dim)

    #  麻雀危险者位置更新
    Temp = range(population_size)
    RandIndex = random.sample(Temp, population_size)
    SDchooseIndex = RandIndex[0:SDNumber]
    for i in range(SDNumber):
        if dominates(Gbest_fitness, particles_F[SDchooseIndex[i]]):
            population[SDchooseIndex[i], :] = Gbest_Location[:] + np.random.randn() * numpy.abs(
                                                population[SDchooseIndex[i], :]-Gbest_Location)
            population[SDchooseIndex[i]] = transform(population[SDchooseIndex[i]], dim)
        elif all(Gbest_fitness == particles_F[SDchooseIndex[i]]):
            K = 2*random.random() - 1
            population[SDchooseIndex[i], :] = population[SDchooseIndex[i], :]+K *\
                                                ((np.abs(population[SDchooseIndex[i], :]-Gworst_Location))
                                                    / (sum(particles_F[SDchooseIndex[i]]-Gworst_fitness)+10E-8))
            for j in range(population.shape[1]):
                if population[SDchooseIndex[i]][j] > 6:  # 速度向量越界检查
                    population[SDchooseIndex[i]][j] = 6
                if population[SDchooseIndex[i]][j] < -6:
                    population[SDchooseIndex[i]][j] = -6
            population[SDchooseIndex[i]] = transform(population[SDchooseIndex[i]], dim)

    t = t + 1
    end = time.time()
    endtime = end + (T - 1 - t) * (end - start)
    endtimeArray = time.localtime(endtime)
    endtimeString = time.strftime("%Y-%m-%d %H:%M:%S", endtimeArray)

    totle_time = totle_time + (end - start)
    print("现在已经到了第", t, "代", ",这一代耗时", end - start, "秒", ",预计结束时间:", endtimeString)

Archive_F1 = (list)(set([tuple(t) for t in Archive_F1]))
# 按照错误率进行升序排序
Archive_F1.sort(key=lambda x: x[0], reverse=False)
Archive_F1 = numpy.array(Archive_F1)

# 去除最后的列表中的重复元素
Archive_F2 = (list)(set([tuple(t) for t in Archive_F2]))
# 按照错误率进行升序排序
Archive_F2.sort(key=lambda x: x[0], reverse=False)
Archive_F2 = numpy.array(Archive_F2)

# 去除最后的列表中的重复元素
Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, population, particles_F)
Archive_F = (list)(set([tuple(t) for t in Archive_F]))
# 按照错误率进行升序排序
Archive_F.sort(key=lambda x: x[0], reverse=False)
Archive_F = numpy.array(Archive_F)

for i in range(0, len(Archive_F1[:, 0])):
    worksheet1.write(i + 1, 0, Archive_F1[i][0])
    worksheet1.write(i + 1, 1, Archive_F1[i][1])
    worksheet1.write(i + 1, 2, Archive_F1[i][2])
for i in range(0, len(Archive_F2[:, 0])):
    worksheet2.write(i + 1, 0, Archive_F2[i][0])
    worksheet2.write(i + 1, 1, Archive_F2[i][1])
    worksheet2.write(i + 1, 2, Archive_F2[i][2])
for i in range(0, len(Archive_F[:, 0])):
    worksheet3.write(i + 1, 0, Archive_F[i][0])
    worksheet3.write(i + 1, 1, Archive_F[i][1])
    worksheet3.write(i + 1, 2, Archive_F[i][2])
worksheet3.write(i + 2, 0, totle_time)
workbook.close()
print("over~")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴刻度负号乱码
plot1, = plt.plot(Archive_F1[:, 0], Archive_F1[:, 1], '^')  # r是红色，o表示圆点
plot2, = plt.plot(Archive_F2[:, 0], Archive_F2[:, 1], 'g*')  # g是绿色，o表示星号
plot3, = plt.plot(Archive_F[:, 0], Archive_F[:, 1], 'ro')  # y是黄色，^表示上三角
# plt.legend([plot1, plot2, plot3, plot4],['WaveformEW_MODA', 'WaveformEW_MOGWO','WaveformEW_MOPSO', 'WaveformEW_NSGAII'])
plt.legend([plot1, plot2, plot3], ['MOHHO_30代', 'MOHHO_60代', 'MOHHO_100代'])

plt.title('MODA')
plt.xlabel('特征数目/总数目')
plt.ylabel('1-Accuracy')
plt.show()