import os
import numpy as np

import matplotlib.pyplot as plt


class GA:
    def __init__(self):

        # 自由设定的变量
        self.population_size = 150
        self.max_iteration = 2000
        self.crossover_probability = 0.75
        self.mutate_probability = 0.2
        self.tournament_size = 8

        # 根据输入文件变更的变量
        self.node_length = 0
        self.distance_metrix = None
        self.init_node_list = []

        # 辅助运算的变量
        self.__distance_metrix = None
        self.__plt_distance = np.zeros(self.max_iteration)

    def read_file(self, file_path):
        """读文件, 计算距离矩阵"""
        if not os.path.isfile(file_path):
            # 文件不存在
            return False

        with open(file_path, "r") as file:
            for line in file:
                if line is not None:
                    if line[0].isdigit():
                        # 数据行读入
                        line = line.strip()
                        line_elements = line.split()
                        # 转为浮点数便于计算
                        self.init_node_list.append((float(line_elements[1]), float(line_elements[2])))

        n = len(self.init_node_list)
        self.node_length = n

        self.__distance_metrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # 求出两点间的欧氏距离
                d = np.linalg.norm(np.array(self.init_node_list[i]) - np.array(self.init_node_list[j]))
                self.__distance_metrix[i, j] = self.__distance_metrix[j, i] = d

        self.distance_metrix = self.__distance_metrix

        return True

    def init_population(self, node_list=None, population_size=None):
        if node_list is None:
            node_list = self.init_node_list
        if population_size is None:
            population_size = self.population_size

        population = []

        for i in range(population_size):
            # 使用排列作为种群个体
            permutation = np.arange(len(node_list))
            # 向种群中加入随机打乱的排列
            np.random.shuffle(permutation)
            population.append(permutation)

        return population

    def fitness(self, single_path, node_list):
        """计算种群适应度"""

        distance = self.__distance_metrix[int(single_path[-1]), int(single_path[0])]
        for i in range(len(node_list) - 1):
            distance += self.__distance_metrix[single_path[i], single_path[i + 1]]

        return 1 / distance

    def select(self, population, node_list):
        """选择操作"""
        # 适应度列表
        fitness_list = np.array([self.fitness(i, node_list) for i in population])

        # 当前种群适应度排序，从大到小
        # np.argsort(-np.array([self.fitness(i, node_list) for i in population]))

        # 选用锦标赛算子
        # 选取优良个体
        select_list = []
        for i in range(self.population_size):
            # 锦标赛算法
            players = np.random.choice(np.arange(self.population_size), size=self.tournament_size, replace=False) # n个随机选手
            best_player = players[np.array([fitness_list[player] for player in players]).argmax()]
            select_list.append(population[best_player])
        return select_list

    def crossover(self, population):
        """染色体交叉"""
        crossover_list = []

        for i in range(0, self.population_size, 2):
            if np.random.random_sample() <= self.crossover_probability:
                child1, child2 = self.crossover_ox_op(population[i], population[i+1])
                crossover_list.append(child1)
                crossover_list.append(child2)
            else:
                crossover_list.append(population[i])
                crossover_list.append(population[i+1])

        return crossover_list

    def crossover_ox_op(self, parent1, parent2):
        """顺序交叉算子"""
        # 初始化
        child1 = np.zeros(self.node_length)
        child2 = np.zeros(self.node_length)

        # 生成两个位点
        pos1, pos2 = np.random.choice(np.arange(self.node_length), size=2, replace=False)
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        # 复制选取的段
        child1[pos1: pos2] = parent1[pos1: pos2]
        child2[pos1: pos2] = parent2[pos1: pos2]

        # 父母去掉已复制的段
        tmp1 = np.delete(parent1, np.where(np.isin(parent1, parent2[pos1: pos2])))
        tmp2 = np.delete(parent2, np.where(np.isin(parent2, parent1[pos1: pos2])))

        # 剩余部分按顺序交换
        child1[0: pos1] = tmp2[0: pos1]
        child2[0: pos1] = tmp1[0: pos1]
        child1[pos2:] = tmp2[pos1:]
        child2[pos2:] = tmp1[pos1:]

        return child1.astype(int), child2.astype(int)

    def mutate(self, population):
        """变异"""
        mutate_list = []

        for i in range(self.population_size):
            if np.random.random_sample() <= self.mutate_probability:
                mutate_list.append(self.mutate_swap_op(population[i]))
            else:
                mutate_list.append(population[i])

        return mutate_list

    def mutate_swap_op(self, individual):
        """反转变异算子"""
        # 生成两个位点
        pos1, pos2 = np.random.choice(np.arange(self.node_length), size=2, replace=False)
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        # 反转
        individual[pos1:pos2] = individual[pos1:pos2][::-1]

        return individual

    def run(self):
        # 记录最优解
        best_individual = None
        best_fitness = 0
        best_iteration = -1

        # 产生初始种群
        population = self.init_population()
        # 开始迭代
        for iteration in range(self.max_iteration):

            # 选择
            select_population = self.select(population, self.init_node_list)
            # 交叉
            crossover_population = self.crossover(select_population)
            # 变异
            mutate_population = self.mutate(crossover_population)
            # 新一代种群
            population = mutate_population

            # 统计
            fitness_list = np.array([self.fitness(i, self.init_node_list) for i in population])
            current_best_index = fitness_list.argmax()

            current_best_individual = population[current_best_index]
            current_best_fitness = fitness_list[current_best_index]

            self.__plt_distance[iteration] = 1 / current_best_fitness

            if current_best_fitness > best_fitness:
                best_individual = current_best_individual
                best_fitness = current_best_fitness
                best_iteration = iteration

            if iteration % 10 == 0:
                print(f'当前第{iteration}轮，当前轮次最优秀个体{1 / current_best_fitness}，所有轮次最优秀个体{1/best_fitness}，出现在第{best_iteration}轮')

            # self.crossover_probability = self.crossover_probability + (0.9-self.crossover_probability)*(iteration/self.max_iteration)
            # self.mutate_probability = self.mutate_probability - (self.mutate_probability - 0.2)*(iteration/self.max_iteration)

        print(f'求得最优解： {1/best_fitness}')

    def show_plot(self):
        x = np.arange(self.max_iteration)
        plt.plot(x, self.__plt_distance)
        plt.ylim(ymin=567)
        plt.show()


if __name__ == "__main__":

    ga = GA()
    ga.read_file('./Data/xqf131.tsp')

    ga.run()

    best = np.array([0, 11, 4, 12, 17, 24, 15, 13, 14, 16, 18, 26, 25, 44, 52, 73, 63, 67, 74, 76, 77, 80, 81, 86, 87, 91, 93, 98, 92, 88, 97, 111, 122, 129, 120, 117, 113, 104, 99, 100, 101, 105, 106, 107, 112, 123, 124, 125, 130, 126, 127, 128, 121, 116, 119, 115, 118, 114, 108, 109, 110, 102, 103, 96, 95, 94, 90, 89, 85, 84, 83, 82, 78, 79, 71, 72, 60, 59, 58, 57, 62, 66, 70, 75, 69, 65, 68, 64, 61, 55, 56, 51, 50, 49, 48, 47, 46, 54, 53, 45, 27, 28, 19, 29, 30, 31, 20, 32, 33, 34, 35, 21, 36, 37, 38, 22, 39, 40, 41, 42, 43, 23, 10, 3, 9, 8, 2, 1, 7, 6, 5])
    print('官方最优解：', 1 / ga.fitness(best, ga.init_node_list))

    ga.show_plot()
