import random
import math
import matplotlib.pyplot as plt  # 用来画图的包
import numpy as np  # numpy是用来做数据处理的包
import pandas as pd  # pandas用来做数据处理的包
from sklearn.model_selection import train_test_split  # 将数据集划分为训练集和测试集   train_test_split表示训练测试和划分
from sklearn.multiclass import OneVsOneClassifier

from sklearn.svm import LinearSVC  # 支持向量机  ？？？？
from sklearn.preprocessing import StandardScaler  # 数据预处理之数据标准化
from sklearn.svm import SVC  # 支持向量机
from sklearn.metrics import accuracy_score  # accuracy_score是准确率
import time
import xlsxwriter  # 进行execl操作  xlsxwriter总之比xlwt好

np.set_printoptions(threshold=np.inf)  # 如果输出的特征矩阵过大会出现省略号的情况，加上这句话就完整输出，不会出现省略号的情况


# 获取所有样本的三个目标函数的值
def get_ObjectFunction(population, dim, x, y, z):
    f = []
    for i in range(population.shape[0]):  # x.shape[0]表示输出矩阵的行数，x.shape[1]表示输出矩阵的列数，这里循环100个个体次
        while (sum(population[i]) < 2):  # （选择的特征最少得有一个）对于全是false 则更新个体
            population[i] = np.random.randint(0, 2, (1, dim))
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


def dominates(x, y):
    if all(x <= y) and any(x < y):
        return True  # x支配y
    else:
        return False


# 更新档案
def updateArchive(Archive_X, Archive_F, population, particles_F):  # 怎么求最里边的那几个数据
    Archive_temp_X = np.vstack((Archive_X, population))  # 纵向合并 136，36
    Archive_temp_F = np.vstack((Archive_F, particles_F))  # 纵向合并 136，3
    o = np.zeros(Archive_temp_F.shape[0])  # 表示矩阵的行数Archive_temp_X.shape[0] 136
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
    pass


def RankingProcess(Archive_F, ArchiveMaxSize, obj_no):  # 排序流程 是对搜索空间划分网格，获得当前档案集的位置
    # 传递过来的是个数组类型
    if len(Archive_F) == 1:  # 如果数组的长度是1
        my_min = [Archive_F[0][0], Archive_F[0][1], Archive_F[0][2]]
        my_max = [Archive_F[0][0], Archive_F[0][1], Archive_F[0][2]]
    else:
        my_min = [min(Archive_F[:, 0]), min(Archive_F[:, 1]), min(Archive_F[:, 2])]  # my_min表示输出每一列的最小值
        my_max = [max(Archive_F[:, 0]), max(Archive_F[:, 1]), max(Archive_F[:, 2])]  # my_max表示输出每一列的最大值

    r = [(my_max[0] - my_min[0]) / 10, (my_max[1] - my_min[1]) / 10, (my_max[2] - my_min[2] / 10)]
    # 最大值-最小值得到一个片段，将纵轴和横轴分成多少份
    ranks = np.zeros(len(Archive_F))
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


def handleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize):  # 处理存档值已经满的情况
    for i in range(len(Archive_F) - ArchiveMaxSize):
        index = RouletteWheelSelection(Archive_mem_ranks)
        Archive_X = np.vstack(
            (Archive_X[0:index], Archive_X[index + 1:Archive_member_no]))  # vstack表示纵向合并  a[0:2],截取矩阵a的第一行到第二行，前闭后开
        Archive_F = np.vstack((Archive_F[0:index], Archive_F[index + 1:Archive_member_no]))  # hstack表示横向合并
        Archive_mem_ranks = np.hstack((Archive_mem_ranks[0:index], Archive_mem_ranks[index + 1:Archive_member_no]))
        Archive_member_no = Archive_member_no - 1

    Archive_X_Chopped = 1.0 * Archive_X
    Archive_F_Chopped = 1.0 * Archive_F
    Archive_mem_ranks_updated = 1.0 * Archive_mem_ranks
    return Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no
    pass


# 轮盘赌机制：被选中的概率与个体在总体中所占比例成正比。
def RouletteWheelSelection(weights):  # eg [1. 2. 1. 2. 4.]
    accumulation = np.cumsum(weights)  # accumulation=[1. 3. 4. 6. 10.]
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


# 下面是主程序

# 在此处更换数据集的名字👇

inputname = 'hypothyroid'  # Zoo 16个特征   Krvskp 36个特征

# 只改上面就可以的黑盒     👆别处不要动
inputdata = 'C:/Users/dell/Desktop/data-cost/' + inputname + ".csv"  # inputdata是数据集的存放地址
inputdata1 = 'C:/Users/dell/Desktop/data-cost/' + inputname + '-cost' + ".csv"

dataset = pd.read_csv(inputdata, header=None)
dataset1 = pd.read_csv(inputdata1, header=None)

workbook = xlsxwriter.Workbook('C:/Users/dell/Desktop/data-cost/' + inputname + '_MODA4' + '.xlsx')
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

# 将非劣解保存到Archive中去
ArchiveMaxSize = 100  # 存档最大值

Archive_X = np.zeros((ArchiveMaxSize, dim))  # 生成一个100行36列的值为0的矩阵  用来存储样本的值，即一条x的值
Archive_F = np.ones((ArchiveMaxSize, obj_no)) * float("inf")  # Archive_F表示100行3列的值为无穷大的矩阵  用来存储三个目标函数的值

Archive_member_no = 0  # 存档数

Archive_F1 = []
Archive_F2 = []
number = []

food_fitness = float("inf") * np.ones(3)  # 有三个目标函数，因此表示三个值为正无穷大
food_pos = np.zeros(dim)  # 食物的位置——>二进制串
enemy_fitness = -float("inf") * np.ones(3)  # 有三个目标函数，因此表表示三个值为负无穷大
enemy_pos = np.zeros(dim)  # 天敌的位置——>二进制串

# 生成一个初始的种群
population = np.random.randint(0, 2, (population_size, dim))  # population表示一个30行36列的0/1矩阵
Delta_population = np.random.randint(0, 2, (population_size, dim))  # 速度值
# print(population)
# print(Delta_population)

totle_time = 0
for t in range(T):  # 注意这里是迭代T次
    start = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    # r=(ub-lb)/4+((ub-lb)*(t/T)*2);   # r表示半径  此算法忽略半径，将整个群体都看成邻居
    w = 0.9 - t * ((0.9 - 0.2) / T)  # 这个小数怎么试呢？？？
    pct = 0.1 - t * ((0.1 - 0) / (T / 2))
    if pct < 0:
        pct = 0
    if t < (3 * T / 4):
        s = pct
        a = pct
        c = pct
        f = 2 * random.random()
        e = pct
    else:
        s = pct / t  # 表示5个操作分离、排队、结盟、寻找猎物和躲避天敌的权重
        a = pct / t
        c = pct / t
        f = 2 * random.random()
        e = pct / t

    # 计算每个个体的两个目标函数的值   particles_F
    particles_F = get_ObjectFunction(population, dim, x, y, z)  # 应该有36行，3列

    for i in range(population.shape[0]):  # ？这还有什么意义吗？下面没用到啊
        if dominates(particles_F[i], food_fitness[0]):  # 更新食物
            food_fitness = particles_F[i]
            food_pos = population[i]
            pass
        if dominates(enemy_fitness[0], particles_F[i]):  # 更新敌人
            enemy_fitness = particles_F[i]
            enemy_pos = population[i]
            pass
        pass

    Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, population, particles_F)
    # 以上都没问题
    if Archive_member_no > ArchiveMaxSize:  # 进行档案集的越界判断
        Archive_mem_ranks = RankingProcess(np.array(Archive_F), ArchiveMaxSize, obj_no)  # 对每个解周围的解进行筛选
        Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = \
            handleFullArchive(np.array(Archive_X), np.array(Archive_F),
                              Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
        pass
    else:
        Archive_mem_ranks = RankingProcess(np.array(Archive_F), ArchiveMaxSize, obj_no)  # 包含每个个体对应的周围解的个数

    if t == 30:
        print("以上完成了{}次迭代".format(t))
        for q in range(len(Archive_F)):  # 30代时，档案集中的最小特征值
            Archive_F1.append(Archive_F[q])
    if t == 60:
        print("以上完成了{}次迭代".format(t))
        for q in range(len(Archive_F)):  # 60代时，档案集中的最小特征值
            Archive_F2.append(Archive_F[q])

    # 选择人口最少的地区的档案馆成员作为食物以提高覆盖率
    index = RouletteWheelSelection(1 / np.array(Archive_mem_ranks))  # 此处是轮盘选择
    print('index(food):', index)
    if index == -1:
        index = 0
    food_fitness = Archive_F[index]
    food_pos = Archive_X[index]

    # 选择人口最多地区的档案馆成员作为敌人提高覆盖率
    index = RouletteWheelSelection(Archive_mem_ranks)  # 此处是轮盘选择
    print('index(enemy):', index)
    if index == -1:
        index = 0
    enemy_fitness = Archive_F[index]
    enemy_pos = Archive_X[index]

    # 以下内容是要更新population了，不考虑邻居，所有的群体都视为邻居
    # 计算 Si、Ai、Ci、Fi、Ei
    for i in range(population.shape[0]):
        # 计算Seperation  Si
        sumS1 = np.zeros(population.shape[1])  # 1*36的矩阵
        Si = np.zeros(population.shape[1])  # 1*36的矩阵
        for j in range(population.shape[0]):
            sumS1 = sumS1 + (population[j] - population[i])
        Si = - sumS1
        # 计算排队Alignment Ai
        sumS2 = np.zeros((1, population.shape[1]))  #
        Ai = np.zeros((1, population.shape[1]))
        for j in range(population.shape[0]):
            sumS2 = sumS2 + Delta_population[j]
        Ai = sumS2 / (population.shape[0])
        # 计算结盟Cohesion Ci
        sumS3 = np.zeros((1, population.shape[1]))
        for j in range(population.shape[0]):
            sumS3 = sumS3 + population[j]
        Ci = sumS3 / (population.shape[0]) - population[i]
        # 计算寻找猎物 Fi
        Fi = food_pos - population[i]
        # 计算躲避天敌 Ei
        Ei = enemy_pos + population[i]
        # 更新公式：
        Delta_population[i] = (s * Si + a * Ai + c * Ci + f * Fi + e * Ei) + w * Delta_population[i]

        # population[i] = population[i]+Delta_population[i]
        pass
    # 根据什么把新的population改成0/1的串  （连续值离散化）
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            if Delta_population[i][j] > 6:  # 速度向量越界检查
                Delta_population[i][j] = 6
            if Delta_population[i][j] < -6:
                Delta_population[i][j] = -6
            # 用的S型的传递函数
            p = random.random()  # 随机生成一个0-1之间的数字
            # s = 1 / (1 + math.exp((-1) * Delta_population[i][j]))
            # 用V型的传递函数  v形传递函数来计算所有解的每个维度的位置改变概率。
            # 根据增量来判断个体的位置是否需要更改，如从0变1，从1变0
            # 都是根据增量/速度来进行位置的二进制映射
            v = math.fabs(Delta_population[i][j] / math.sqrt(Delta_population[i][j] * Delta_population[i][j] + 1))
            if (p < v):
                if population[i][j] == 1:
                    population[i][j] = 0
                else:
                    population[i][j] = 1
            else:
                population[i][j] = population[i][j]
            pass
    pass
    print("At the iteration {} there are {} non-dominated solutions in the archive".format(t, Archive_member_no))

    # for g in range(len(Archive_F)):
    #     Archive_F_all.append(Archive_F[g])
    #     Archive_X_all.append(Archive_X[g])

    end = time.time()
    endtime = end + (T - 1 - t) * (end - start)
    endtimeArray = time.localtime(endtime)
    endtimeString = time.strftime("%Y-%m-%d %H:%M:%S", endtimeArray)

    totle_time = totle_time + (end - start)
    print("现在已经到了第", t, "代", ",这一代耗时", end - start, "秒", ",预计结束时间:", endtimeString)

Archive_F1 = (list)(set([tuple(t) for t in Archive_F1]))
# 按照错误率进行升序排序
Archive_F1.sort(key=lambda x: x[0], reverse=False)
Archive_F1 = np.array(Archive_F1)

# 去除最后的列表中的重复元素
Archive_F2 = (list)(set([tuple(t) for t in Archive_F2]))
# 按照错误率进行升序排序
Archive_F2.sort(key=lambda x: x[0], reverse=False)
Archive_F2 = np.array(Archive_F2)

# 去除最后的列表中的重复元素
Archive_F = (list)(set([tuple(t) for t in Archive_F]))
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
worksheet3.write(i + 2, 0, totle_time)
workbook.close()
print("over~")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴刻度负号乱码
plot1, = plt.plot(Archive_F1[:, 0], Archive_F1[:, 2], '^')  # r是红色，o表示圆点
plot2, = plt.plot(Archive_F2[:, 0], Archive_F2[:, 2], 'g*')  # g是绿色，o表示星号
plot3, = plt.plot(Archive_F[:, 0], Archive_F[:, 2], 'ro')  # y是黄色，^表示上三角
# plt.legend([plot1, plot2, plot3, plot4],['WaveformEW_MODA', 'WaveformEW_MOGWO','WaveformEW_MOPSO', 'WaveformEW_NSGAII'])
plt.legend([plot1, plot2, plot3], [inputname + '_MODA_30代', inputname + '_MODA_60代', inputname + '_MODA_100代'])

plt.title('MODA')
plt.xlabel('错误率')
plt.ylabel('成本值')
plt.show()
