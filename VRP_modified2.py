import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

from VRP import My_GA, crossover_individual_impl, chromosome_impl, mutation_individual_impl
import time
from necessary_parameter import *
import utill


def get_distance(mode, m1, m2):
    global count_mode3
    x1, y1 = m1
    x2, y2 = m2
    if mode == 1:  # taxicap
        return abs(x2 - x1) + abs(y2 - y1)
    if mode == 2:  # 유클리드
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if mode == 3:  # 변형
        #time.sleep(0.3)
        count_mode3 += 1
        # 거리 dm, 평균 속력 v km/h 일때 시간은 0.06d / v 분이므로 v = 40일때 다음과 같다. 단 거리식은 변형해서 사용
        return ((4 * (x2 - x1)) ** 4 + (y2 - y1) ** 4) ** 0.25 * 0.06 / 40
    if mode == 4:
        count_mode3 += 1
        return ((((x2 - x1) ** 4 + 3 * (y2 - y1) ** 4) ** 0.25 + 2 * abs(x2 - x1) + abs(y2 - y1)) / 2) * 0.06 / 30


def get_fitness(chromosome):
    """
    fitness_function
    :param chromosome:
    :return: fitness(float), 논문의 fitness 가져옴
    """
    global time_arr3
    global count_mode3

    temp_ga = My_GA()
    temp_ga.set_x(chromosome)
    temp_ga.set_s(s)
    temp_ga.setMap(map_)
    temp_ga.set_demend(D)

    if ga.current_generation == 0 and fitness_mode == 0:
        temp_ga.setTime(mode=OPTIONAL_DISTANCE_MODE)
        time_arr3 = temp_ga.time_arr
        count_mode3 += len(np.where(time_arr3 != 0)[0])

    if not temp_ga.check_condition: return 9999999

    """
    fitness_mode = 0 : 방법1
    fitness_mode = n : 방법 2, n = interval
    fitness_mode = -n : 방법 3, n = benchmark
    """

    fitness = 0
    if fitness_mode > 0:
        for m in range(1, M+1):
            if ga.current_generation % fitness_mode == 0 and ga.current_generation > 0:
                ij_list = list(zip(np.where(temp_ga.x[:, :, m] == 1)[0], np.where(temp_ga.x[:, :, m] == 1)[1]))
                no_time_list = list(zip(np.where(time_arr3 == 0)[0], np.where(time_arr3 == 0)[1]))
                for tup in ij_list:
                    if tup in no_time_list:
                        i, j = tup[0], tup[1]
                        d = get_distance(OPTIONAL_DISTANCE_MODE, ga.Map[i], ga.Map[j])
                        time_arr3[i][j] = d
                        time_arr3[j][i] = d
                fitness += np.sum(time_arr3 * temp_ga.x[:, :, m])

            else:
                fitness += np.sum(ga.time_arr * temp_ga.x[:, :, m])
        if ga.current_generation % fitness_mode == 0 and ga.current_generation > 0: fitness += np.sum(temp_ga.s)

    elif fitness_mode < 0:
        fm = fitness_mode * -1
        for m in range(1, M+1):
            if ga.current_generation >= fm:
                ij_list = list(zip(np.where(temp_ga.x[:, :, m] == 1)[0], np.where(temp_ga.x[:, :, m] == 1)[1]))
                no_time_list = list(zip(np.where(time_arr3 == 0)[0], np.where(time_arr3 == 0)[1]))
                for tup in ij_list:
                    if tup in no_time_list:
                        i, j = tup[0], tup[1]
                        d = get_distance(OPTIONAL_DISTANCE_MODE, ga.Map[i], ga.Map[j])
                        time_arr3[i][j] = d
                        time_arr3[j][i] = d
                fitness += np.sum(time_arr3 * temp_ga.x[:, :, m])
                fitness += np.sum(temp_ga.s)

            else:
                fitness += np.sum(ga.time_arr * temp_ga.x[:, :, m])

        if ga.current_generation >= fm: fitness += np.sum(temp_ga.s)
    elif fitness_mode == 0:
        for m in range(1, M + 1):
            fitness += np.sum(time_arr3 * temp_ga.x[:, :, m])
            fitness += np.sum(temp_ga.s)

    return fitness


class My_GA2(My_GA):
    def pre_setting(self):
        self.chromosome_length = N * 2
        self.population_size = 20
        self.parent_ratio = 0.3
        self.chromosome_impl = chromosome_impl
        self.crossover_individual_impl = crossover_individual_impl
        self.target_fitness_type = 'min'
        self.parent_selection_impl = utill.roulette
        self.fitness_function_impl = get_fitness
        self.mutation_individual_impl = mutation_individual_impl
        self.selection_probability = 0
        self.set_s(s)
        self.setMap(map_)
        self.set_demend(D)
        self.setTime(mode=BASIC_DISTANCE_MODE)
        self.generation_goal = 500
        self.chromosome_mutation_rate = 0.3

def prior_condition_check():
    print('=' * 75)
    print("SAVE_FILE :", SAVE_FILE)
    print("N :", N, "\tM :", M, "\tC :", C)
    print("map_name :", map_name)
    print("BASIC_DISTANCE_MODE :", BASIC_DISTANCE_MODE)
    print("OPTIONAL_DISTANCE_MODE :", OPTIONAL_DISTANCE_MODE)
    print("Please check get_distance func")
    description = "N{0} M{1} C{2} {3} BASIC{4} OPTIONAL{5}".format(N, M, C, map_name, BASIC_DISTANCE_MODE,
                                                                   OPTIONAL_DISTANCE_MODE)
    description += " method3, generation_goal = 500, population_size = 20"
    description += "\n0번인덱스 : description, 1번 인덱스 : 걸린 시간 리스트(float), 2번 인덱스 : 간격에 따른 최적"
    description += " fitness list 3번 인덱스 : count_mode3_list"
    print(description)
    check = input("Run? (Y/N) : ")
    if check == 'Y' or check == 'y':
        return description
    else: raise Exception("repair parameter")

SAVE_FILE = "result/Rtype_method4.pickle"
time_arr3 = np.zeros((N+1, N+1))
count_mode3 = 0
BASIC_DISTANCE_MODE = 2
OPTIONAL_DISTANCE_MODE = 3

time_list = []
fitness_list = []
count_mode3_list = []
before_fitness = 0
small_count = 0
if __name__ == "__main__":
    description = prior_condition_check()

    count_mode3 = 0
    time_arr3 = np.zeros((N + 1, N + 1))
    fitness_mode = 1001
    ga = My_GA2()
    ga.pre_setting()
    t1 = time.time()
    while ga.active():
        ga.evolve(1)
        current_fitness = ga.population[0].fitness
        df = abs(current_fitness - before_fitness)
        if df == 0:
            small_count += 1
            if small_count > 3 and fitness_mode > 1:
                fitness_mode = 1
                small_count = 0
            elif small_count > 100 and fitness_mode <= 1:
                break
        fitness_list.append(current_fitness)
        count_mode3_list.append(count_mode3)
        before_fitness = current_fitness
    t2 = time.time()

    if map_name == "data/data_Rtype.pickle":
        delta_time = t2 - t1 + 0.3 * count_mode3
    elif map_name == 'data/data_Ctype.pickle':
        delta_time = utill.caculate_running_time(time_arr3, ga.time_arr)

    print('time :', delta_time)
    print('fitness :', ga.population[0].fitness)
    print('count_mode3 :', count_mode3)
    print('-' * 75)

    import pickle

    if input('save? :') == 'Y':
        with open(SAVE_FILE, "wb") as fw:
            pickle.dump([description, time_list, fitness_list, count_mode3_list], fw)
