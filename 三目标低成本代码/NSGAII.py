# encoding:utf-8
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import xlsxwriter
from toolz import random_sample

# 这个应该是NSGA-II的多目标特征选择


def delete(list):  # 改变种群中的相同个体，目的应该是增加多样性
    for i in range(list.shape[0]):
        for j in range(list.shape[0]):
            if (all(list[i] == list[j])and i!=j):

                print('相同的', i, j, all(list[i] == list[j]))

                list[j]= np.random.randint(0, 2, (1, chromosome_length))
                print('nest', j, '改为', list[j])
    return(list)


def transform(population,chromosome_length):   # 该函数是对位置进行二进制映射
    population_fit = population * 1
    p = random.random()
    for i in range(population.shape[0]):  # 行数
        for j in range(population.shape[1]):  # 列数
            a = 1 / (1 + np.exp(-population[i][j]))  # sigmoid 函数 对位置进行二进制映射
            if a > 0.5:
                population_fit[i][j] = 1
            if a < 0.5:
                population_fit[i][j] = 0
        while (sum(population_fit[i]) < 2):  # 有全False时重新生成个体-
            b = np.random.rand(1, chromosome_length)
            population_fit[i] = b * 1
            for j in range(population.shape[1]):
                a = 1 / (1 + np.exp(-population_fit[i][j]))
                if a > 0.5:
                    population_fit[i][j] = 1
                if a < 0.5:
                    population_fit[i][j] = 0
    return (population_fit)


def fast_non_dominated_sort(values1, values2, values3):
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
    print('front', front)
    return front


def crowding_distance(values1, values2, values3, front):  # 适应度函数值和非支配解集每一层 front[i]
    distance = [0 for i in range(0, len(front))]  # 初始为0
    # 初始化个体间的拥挤距离
    sorted1 = sort_by_values(front, values1[:])  # 返回list[0]的个数
    sorted2 = sort_by_values(front, values2[:])
    sorted3 = sort_by_values(front, values3[:])
    # 基于目标函数1和目标函数2对已经划分好层级的种群排序
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values3[sorted3[k + 1]] - values3[sorted3[k - 1]]) / (max(values3) - min(values3))
    return distance


def sort_by_values(list1, values):  # 非支配解集和适应度函数值
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        # 当结果长度不等于初始长度时，继续循环
        if index_of(min(values), values) in list1:  # 最小的适应度函数值的索引在非支配解中
            # 标定值中最小值在目标列表中时
            sorted_list.append(index_of(min(values), values))
        #     将标定值的最小值的索引追加到结果列表后面
        values[index_of(min(values), values)] = math.inf
    #     将标定值的最小值置为无穷小,即删除原来的最小值,移向下一个
    #     infinited
    # print(sorted_list)
    return sorted_list


def index_of(a, list):  # 查找索引，这个可能应该有对应的库函数
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


def get_fitness(population, chromosome_length, x, y, z):

    fitness1 = []
    fitness2 = []
    fitness3 = []
    populationrecord = []

    for i in range(population.shape[0]):
        while (sum(population[i]) < 2):  # （选择的特征最少得有一个）对于全是false 则更新个体
            population[i] = np.random.randint(0, 2, (1, chromosome_length))
        # print('ceshi',population_fit.shape[0])
        column_use = (population[i] == 1)  # 选择个体中为1的特征值
        #        q=[int(i) for i in column_use]
        #        s=sum(q)
        x_test = x.columns[column_use]  # 得到值为true的所有列号，这些列号形成一个列表
        clf = OneVsOneClassifier(SVC(C=1.0, random_state=20, kernel='rbf'))
        X_train, X_test, y_train, y_test = train_test_split(x[x_test], y, test_size=0.7, random_state=7)
        clf.fit(X_train, y_train)
        fitness_1 = 1 - accuracy_score(y_test, clf.predict(X_test))  # 正确率
        # accuracyrecord.append(fitness_1)

        # 求特征向量的个数
        number.append(sum(column_use))
        fitness_2 = 1.0 * (sum(column_use)) / chromosome_length

        # 求成本之和
        z_test = z.columns[column_use]
        costSum = 0
        for i in z_test:
            costSum += z[i]

        # fitness_final = 0.99 * (1 - fitness_1) + 0.01 * fitness_2
        fitness1.append(fitness_1)
        fitness2.append(fitness_2)  # 此处完全以精准度作为适应度函数值？ 这应该是选择特征值的个数
        if isinstance(costSum, int):
            fitness3.append(costSum)  # 成本最小值
        else:
            fitness3.append(costSum[0])
        populationrecord.append(column_use)  # 存入每个个体选择的特征情况，包含列的ture和false
        # index = fitness.index(min(fitness))

    return populationrecord, fitness1, fitness2, fitness3    # 个体记录 精度  适应度函数的值


# 下面是主程序
# 在此处更换数据集的名字👇

inputname = 'heart'

# 只改上面就可以的黑盒    👆 别处不要动
inputdata = 'C:/Users/dell/Desktop/data-cost/' + inputname + ".csv"

inputdata1 = 'C:/Users/dell/Desktop/data-cost/' + inputname + '-cost' + ".csv"

dataset = pd.read_csv(inputdata, header=None)
dataset1 = pd.read_csv(inputdata1, header=None)

workbook = xlsxwriter.Workbook('C:/Users/dell/Desktop/data-cost/' + inputname + '_NSGAII' + '.xlsx')
worksheet1 = workbook.add_worksheet("30代")
worksheet2 = workbook.add_worksheet("60代")
worksheet3 = workbook.add_worksheet("100代")
ColName = ["f1（错误率）", 'f2（特征子集/总特征）', 'f3(最小成本)']
for i in range(len(ColName)):
    worksheet1.write(0, i, ColName[i])
    worksheet2.write(0, i, ColName[i])
    worksheet3.write(0, i, ColName[i])

x = dataset.iloc[:, 0:-1]  # 特征
x = pd.DataFrame(StandardScaler().fit_transform(x))  # 利用支持向量机最好进行数据归一化预处理
y = dataset.iloc[:, -1]  # 标签

z = dataset1.iloc[:]  # 成本

population_size = 30

number = []
chromosome_length = x.shape[1]

AP = 0.1  # 意识概率
fl = 2  # 飞行长度

Archive_F = []
Archive_F1 = []
Archive_F2 = []

gen_no = 0  # 迭代次数
max_gen = 100  # 最大迭代次数
population = 2*np.random.random((population_size, chromosome_length))-1
population = transform(population, chromosome_length)  # 对随机的个体进行二进制映射
while gen_no < max_gen:
    start = time.time()
    population = delete(population)  # 删除其中近似的个体

    population_mem = population*1

    pop, fitness1, fitness2, fitness3 = get_fitness(population, chromosome_length, x, y, z)
    function1_values = [fitness1[i] for i in range(0, population_size)]  # 与fitness1有何区别？

    function2_values = [fitness2[i] for i in range(0, population_size)]

    function3_values = [fitness3[i] for i in range(0, population_size)]

    non_dominated_sorted_solution = fast_non_dominated_sort(
        function1_values[:], function2_values[:], function3_values[:])
    # 种群之间进行快速非支配性排序,得到非支配性排序集合
    print("The best front for Generation number ", gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:  # 第一级的非支配解
        print((population[valuez]), end=" ")
    print("\n")
    crowding_distance_values = []
    # 计算非支配集合中每个个体的拥挤度
    for i in range(0, len(non_dominated_sorted_solution)):  # 二维的话，返回行的个数
        crowding_distance_values.append(
            crowding_distance(function1_values[:], function2_values[:], function3_values[:], non_dominated_sorted_solution[i][:]))
    population2 = population[:]

    if gen_no == 0:
        print("以上完成了{}次迭代".format(gen_no))
        for q in non_dominated_sorted_solution[0]:  # 30代时，档案集中的最小特征值
            Archive_F1.append([fitness1[q], fitness2[q], fitness3[q]])
    if gen_no == 60:
        print("以上完成了{}次迭代".format(gen_no))
        for q in range(len(non_dominated_sorted_solution[0])):  # 60代时，档案集中的最小特征值
            Archive_F2.append([fitness1[q], fitness2[q], fitness3[q]])

    chase = random.randint(0, population.shape[0] - 1)  # 随机的个体：0-个体数目
    # 更新位置
    for i in range(population.shape[0]):
        if np.random.rand() > AP:
            population[i] = population[i] + fl * np.random.rand() * (population_mem[chase] - population[i])
        else:
            population[i] = np.random.rand(1, chromosome_length)  # 返回一个1行18列的0-1中的随机值， 相当于修改了个体

        #      population[i,:] = (population[i,:]-np.min(population[i]))/(np.max(population[i])-np.min(population[i]))
    population = transform(population, chromosome_length)

    solution2 = np.vstack((population_mem, population))  # 原种群和新种群进行合并

    pop, fitness1_1, fitness2_1, fitness3_1 = get_fitness(solution2, chromosome_length, x, y, z)  # pop 每个个体的特征选择情况

    function1_values2 = [fitness1_1[i] for i in range(0, solution2.shape[0])]

    function2_values2 = [fitness2_1[i] for i in range(0, solution2.shape[0])]

    function3_values2 = [fitness3_1[i] for i in range(0, solution2.shape[0])]

    non_dominated_sorted_solution2 = fast_non_dominated_sort(
        function1_values2[:], function2_values2[:], function3_values2[:])

    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(
            crowding_distance(function1_values2[:], function2_values2[:], function3_values2[:], non_dominated_sorted_solution2[i][:]))
    # print(4, crowding_distance_values2)

    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [
            index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
            range(0, len(non_dominated_sorted_solution2[i]))]  # 2_1就是解中对应的索引
        # 排序
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])  # 根据拥挤距离进行排序
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                 range(0, len(non_dominated_sorted_solution2[i]))]  # 得到排序后的个体序号
        front.reverse()  # 把拥挤距离大的放在前面

        for value in front:
            new_solution.append(value)
            if len(new_solution) == population_size:
                break
        if len(new_solution) == population_size:
            break
    # 得到原种群和新种群的混合子代
    # print('子代3', solution2)
    print('new_solution', new_solution)  # 保存留下的个体的序号

    nest3 = 2 * np.random.random((population_size, chromosome_length)) - 1
    for i in range(population_size):  # 建立新种群个体
        # print(i,solution2[new_solution[i]])
        # for j in range(nest_size):
        nest3[i, :] = solution2[new_solution[i], :]

    population = nest3

    gen_no = gen_no + 1
    end = time.time()
    endtime = end + (299 - gen_no) * (end - start)
    endtimeArray = time.localtime(endtime)
    endtimeString = time.strftime("%Y-%m-%d %H:%M:%S", endtimeArray)
    print("现在已经到了第", gen_no, "代", ",这一代耗时", end - start, "秒", ",预计结束时间:", endtimeString)

# print('function1_values', function1_values)
# print('function2_values', function2_values)

population = delete(population)
pop, fitness1, fitness2, fitness3 = get_fitness(population, chromosome_length, x, y, z)
non_dominated_sorted_solution = fast_non_dominated_sort(fitness1[:], fitness2[:], fitness3[:])
for q in non_dominated_sorted_solution[0]:  # 30代时，档案集中的最小特征值
    Archive_F.append([fitness1[q], fitness2[q], fitness3[q]])

Archive_F1 = (list)(set([tuple(gen_no) for gen_no in Archive_F1]))
# 按照错误率进行升序排序
Archive_F1.sort(key=lambda x: x[0], reverse=False)
Archive_F1 = np.array(Archive_F1)

# 去除最后的列表中的重复元素
Archive_F2 = (list)(set([tuple(gen_no) for gen_no in Archive_F2]))
# 按照错误率进行升序排序
Archive_F2.sort(key=lambda x: x[0], reverse=False)
Archive_F2 = np.array(Archive_F2)

# 去除最后的列表中的重复元素
Archive_F = (list)(set([tuple(gen_no) for gen_no in Archive_F]))
# 按照错误率进行升序排序
Archive_F.sort(key=lambda x: x[0], reverse=False)
Archive_F = np.array(Archive_F)

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
workbook.close()
print("over~")
print("over~")













