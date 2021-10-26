import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)

from VRP import My_GA, crossover_individual_impl, chromosome_impl, mutation_individual_impl
import time
from necessary_parameter import *
import utill


BASIC_DISTNACE_MODE = 2
OPTIONAL_DISTANCE_MODE = 3
"""
mode = 0 : time_array를 전부 가져온 다음 사용
mode = n : time_array를 generation % fitness_mode == 0일때마다 업테이트 하여 사용
mode = -n : generation이 n보다 크면 mode 바꿈
"""
count_mode3= 0

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
        return ((4* (x2 - x1)) ** 4 + 3 * (y2 - y1) ** 4) ** 0.25

time_arr3 = np.zeros((N+1, N+1))


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

            else:
                fitness += np.sum(ga.time_arr * temp_ga.x[:, :, m])

    elif fitness_mode == 0:
        for m in range(1, M + 1):
            fitness += np.sum(time_arr3 * temp_ga.x[:, :, m])




    fitness += np.sum(temp_ga.s)
    return fitness

class My_GA2(My_GA):
    def pre_setting(self):
        self.chromosome_length = N * 2
        self.population_size = 10
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
        self.setTime(mode=BASIC_DISTNACE_MODE)
        self.generation_goal = 1000
        self.chromosome_mutation_rate = 0.3

time_list = []
fitness_list = []
count_mode3_list = []
small_count= 0
if __name__ == "__main__":
    count_mode3 = 0
    time_arr3 = np.zeros((N + 1, N + 1))
    fitness_mode = 1001
    ga = My_GA2()
    ga.pre_setting()
    t1 = time.time()
    while ga.active():
        ga.evolve(1)
        best_fitness = ga.population[0].fitness
        second_fitness = ga.population[1].fitness
        df = abs(best_fitness - second_fitness)
        if df < 0.1:
            small_count += 1
            if small_count > 10:
                if fitness_mode > 1:
                    fitness_mode = 1
                else:
                    break
    t2 = time.time()
    fitness_mode = 1
    for chromosome in ga.population:
        chromosome.fitness = ga.fitness_function_impl(chromosome)
    """while ga.active():
        # Evolve only a certain number of generations
        ga.evolve(1)
        fitness_mode = 1
        for chromosome in ga.population:
            chromosome.fitness = ga.fitness_function_impl(chromosome)
        fitness_mode = MODE
        # Print the current generation
        if ga.current_generation % 100 == 0:
            ga.print_generation()
            # Print the best chromosome from that generations population
            ga.print_best_chromosome()
            # To divide the print to make it easier to look at
            print('-' * 75)
    """
    time_list.append(t2 - t1 + count_mode3 * 0.3)
    fitness_list.append(ga.population[0].fitness)
    count_mode3_list.append(count_mode3)
    ga.graph.lowest_value_chromosome()
    ga.graph.show()
    print('time :', t2 - t1 + count_mode3 * 0.3)
    print('fitness :', ga.population[0].fitness)
    print('count_mode3 :', count_mode3)
    print('-' * 75)

    import pickle

    with open("result_before/Rtype_method5.pickle", "wb") as fw:
        pickle.dump([time_list, fitness_list, count_mode3_list], fw)

