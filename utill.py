import random
import numpy as np
seed = 42

random.seed(seed)
np.random.seed(seed)

from necessary_parameter import *
from EasyGA.parent.Parent import _check_selection_probability, _ensure_sorted,\
    _check_positive_fitness, _compute_parent_amount

seed = 42

random.seed(seed)
np.random.seed(seed)

def get_fitness(chromosome):
    """
    fitness_function
    :param chromosome:
    :return: fitness(float), 논문의 Z 가져옴
    """
    from VRP import My_GA

    temp_ga = My_GA()
    temp_ga.set_x(chromosome)
    temp_ga.set_s(s)
    temp_ga.setMap(map_)
    temp_ga.set_demend(D)
    temp_ga.setTime(mode=time_mode)

    if not temp_ga.check_condition(): return 9999999  # float('inf') - float('inf') = nan이므로 적절히 큰 수 선택

    fitness = 0
    for m in range(1, M+1):
        fitness += np.sum(temp_ga.time_arr * temp_ga.x[:, :, m])
    fitness += np.sum(temp_ga.s)

    return fitness


@_check_selection_probability
@_ensure_sorted
@_check_positive_fitness
@_compute_parent_amount
def roulette(ga, parent_amount):

    costTofitenss(ga)
    # The sum of all the fitnessess in a population
    fitness_sum = sum(
        ga.real_fitness[index]
        for index
        in range(len(ga.population))
    )

    # A list of ranges that represent the probability of a chromosome getting chosen
    probability = [ga.selection_probability]

    # The chance of being selected increases incrementally
    for index in range(len(ga.population)):
        probability.append(probability[-1]+ga.real_fitness[index]/fitness_sum)

    probability = probability[1:]

    # Loops until it reaches a desired mating pool size
    while len(ga.population.mating_pool) < parent_amount:

        # Spin the roulette
        rand_number = random.random()

        # Find where the roulette landed.
        for index in range(len(probability)):
            if probability[index] >= rand_number:
                ga.population.set_parent(index)
                break

def costTofitenss(ga, k=3):
    """
    현재 클래스 내부의 fitness는 총 걸린시간으로 줄이는걸 목표로 잡아야 한다면
    이 함수를 통해 선택압을 기반으로 한 최대화해야하는 fitness로 변환
    :return: x (ga.real_fitenss에 저장)
    """
    if k < 1: raise Exception("적절하지 않은 선택압")
    cost_list = []
    for chromosome in ga.population:
        cost_list.append(chromosome.fitness)
    max_cost, min_cost = max(cost_list), min(cost_list)
    ga.real_fitness = list(map(lambda c: (max_cost - c) + (max_cost - min_cost) / (k-1), cost_list))
    if ga.real_fitness == [0] * len(ga.real_fitness):
        ga.real_fitness = [1] * len(ga.real_fitness)


def caculate_running_time(time_arr3, distance_arr, k=10 ** (-4)):
    """
    첫번째 실험에서 거리에 상관없이 계산하였다면, 계산한 거리에 비례하는 계산 소요 시간을 구해주는 함수
    :param time_arr3: VRP_modified에서 쓰는 time_arr3
    :param distance_arr: set_time(2)의 결과 (또는 set_time(1)
    :param k: 비례상수, 10^-4이면 1km 계산하는데 0.1초 걸림
    :return: time_arr3를 채우는데 걸리는 예상 시간을 return함
    """
    temp_arr = np.zeros((distance_arr.shape[0], distance_arr.shape[1]))
    temp_arr[np.where(time_arr3 != 0)] = 1
    sum_ = np.sum(temp_arr * distance_arr)
    return sum_ * k / 2  # 대칭행렬이므로 /2
