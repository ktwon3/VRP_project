import random
import numpy as np

seed = 42

random.seed(seed)
np.random.seed(seed)

from EasyGA import *
import utill
from utill import *
from check_condition import *
from necessary_parameter import *
from EasyGA.decorators import _check_weight
from EasyGA.mutation.Mutation import _check_gene_mutation_rate, _loop_random_mutations, _reset_fitness


# Create the Genetic algorithm
class My_GA(GA):
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
        self.setTime(mode=time_mode)
        # print(ga.time_arr)
        # print(ga.x)
        self.generation_goal = 1000
        self.chromosome_mutation_rate = 0.3

    def setMap(self, m):
        """
        Map 지정
        """
        if len(m) == N + 1:
            self.Map = m
        else:
            raise Exception("초깃값 오류 (map)")

    def setTime(self, mode, time_arr=-1):
        """
        :param mode:  mode 0: 직접 time 지정, mode 1이상 : get_distance(mode)에 비례한 time 생성
        :param time_arr: mode 0일때 time_arr 지정
        :return: arr[i][j]가 i지점에서 j지점 까지 갈때 시간인 ndarray
        """

        if mode == 0:  # time이 정해짐
            if time_arr == -1: raise Exception("time_arr didn't setting!")
            if time_arr.shape[0] != N + 1: raise Exception("time_arr's size error!")
            self.time_arr = time_arr
        else:  # 유클리드 거리가 time
            time_arr = np.zeros((len(self.Map), len(self.Map)))
            if len(time_arr) <= 1: raise Exception('맵 크기 오류')
            for i in range(1, len(time_arr)):
                for j in range(0, i):
                    d = get_distance(mode, self.Map[i], self.Map[j])
                    time_arr[i][j] = d
                    time_arr[j][i] = d
            self.time_arr = time_arr

    def set_demend(self, d):
        if len(d) == N + 1 and d[0] == 0:
            self.D = d  # 차고지 고려 (차고지의 수요량은 0)
        else:
            raise Exception("초깃값 오류(수요량)")

    def set_x(self, chromosome):
        """
        :param chromosome: int면 chromosome번째 chromosome을 가져오고, 리스트면 그대로 사용
        :return: chromosome에 대한 x 행렬 반환
        """
        if type(chromosome) != type([1]): chromosome = chromosome = chromosome.gene_value_list
        x = np.zeros((N + 1, N + 1, M + 1))  # (N,N,0)은 빈 행렬, 차량이 1,2,...,M으로 정의했기에
        priority, vehicle_nums = chromosome[:N], chromosome[N:]
        dic = dict(zip(priority, vehicle_nums))

        # [[0,1,2,...,0], [0,3,4,...,0] ... ] 와 같이 경로를 바로 나타내는 리스트로 변환
        path = [[0] for i in range(M + 1)]  # 차량을 1,2,...,M으로 정의했기에 M+1로
        for i in range(1, N + 1):
            try:
                vehicle = dic[i]
            except:
                print(dic)
                raise Exception('asdf')

            try:
                path[vehicle].append(priority.index(i) + 1)
            except:
                print('a')
                raise Exception('asdf')
        for i in range(len(path)):
            path[i].append(0)

        # path를 x행렬로 변환
        for vehicle in range(1, M + 1):
            path_one = path[vehicle]
            if path_one == [0, 0]: continue
            for inx in range(1, len(path_one)):
                j = path_one[inx]
                i = path_one[inx - 1]
                x[i][j][vehicle] = 1
        self.x = x

    def check_condition(self) -> bool:
        """
        하나의 chromosome에 대해 제약 조건을 check함
        :return: 제약 조건 만족시 True, 아닐시 False
        """
        return bool(check_one_visit(self) * check_continuous(self) * check_volume(self) *
                    check_garage(self) * check_nomove(self))

    def set_s(self, s):
        if s[0] == 0 and len(s) == N + 1:
            self.s = np.array(s)
        else:
            raise Exception("초깃값 오류 (s)")

@_check_weight
def crossover_individual_impl(ga, parent_1, parent_2, *, weight=0.5):
    # get gene values
    priority1, vehicle_numbers1 = parent_1[:N], parent_1[N:]
    priority2, vehicle_numbers2 = parent_2[:N], parent_2[N:]

    # 순서교차
    if len(priority1) != len(priority2): raise Exception("부모 길이 다름 오류")
    new_priority1, new_priority2 = [0 for i in range(len(priority1))], [0 for i in range(len(priority2))]
    parent_length = len(priority1)
    swap_index1 = random.randint(1, parent_length - 2)
    swap_index2 = random.randint(swap_index1 + 1, parent_length - 1)

    p1, p2 = priority1[swap_index1:swap_index2], priority2[swap_index1:swap_index2]
    new_priority1[swap_index1:swap_index2] = p2
    new_priority2[swap_index1:swap_index2] = p1

    index1, index2 = swap_index2 - parent_length, swap_index2 - parent_length
    for index in range(swap_index2 - parent_length, swap_index2 + 1):
        if priority1[index] not in p2 and index1 < swap_index1:
            new_priority1[index1] = priority1[index]
            index1 += 1
        if priority2[index] not in p1 and index2 < swap_index1:
            new_priority2[index2] = priority2[index]
            index2 += 1

    # 2점 교차
    if len(vehicle_numbers1) != len(vehicle_numbers2): raise Exception("부모 길이 다름 오류")
    new_vehicle_numbers1, new_vehicle_numbers2 = [0 for i in range(len(vehicle_numbers1))], \
                                                 [0 for i in range(len(vehicle_numbers2))]
    parent_length = len(priority1)
    swap_index1 = random.randint(1, parent_length - 2)
    swap_index2 = random.randint(swap_index1 + 1, parent_length - 1)

    new_vehicle_numbers1[swap_index1:swap_index2] = vehicle_numbers2[swap_index1:swap_index2]
    new_vehicle_numbers1[:swap_index1] = vehicle_numbers1[:swap_index1]
    new_vehicle_numbers1[swap_index2:] = vehicle_numbers1[swap_index2:]

    new_vehicle_numbers2[swap_index1:swap_index2] = vehicle_numbers1[swap_index1:swap_index2]
    new_vehicle_numbers2[:swap_index1] = vehicle_numbers2[:swap_index1]
    new_vehicle_numbers2[swap_index2:] = vehicle_numbers2[swap_index2:]

    ga.population.add_child(new_priority1 + new_vehicle_numbers1)
    ga.population.add_child(new_priority2 + new_vehicle_numbers2)

def chromosome_impl(ga):
    while True:
        l1 = random.sample([i for i in range(1, N + 1)], N)  # priority
        l2 = [random.randint(1, M) for i in range(N)]  # vehicle_nums
        l12 = l1 + l2
        temp_ga = My_GA()
        temp_ga.set_x(l12)
        temp_ga.set_demend(D)
        if check_volume(temp_ga): break
    return l12


def get_distance(mode, m1, m2):
    x1, y1 = m1
    x2, y2 = m2
    if mode == 1:  # taxicap
        return abs(x2 - x1) + abs(y2 - y1)
    if mode == 2:  # 유클리드
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if mode == 3:  # 변형
        #time.sleep(0.3)
        # 거리 dm, 평균 속력 v km/h 일때 시간은 0.06d / v 분이므로 v = 40일때 다음과 같다. 단 거리식은 변형해서 사용
        return ((4 * (x2 - x1)) ** 4 + (y2 - y1) ** 4) ** 0.25 * 0.06 / 40
    if mode == 4:
        return ((((x2 - x1) ** 4 + 3 * (y2 - y1) ** 4) ** 0.25 + 2 * abs(x2 - x1) + abs(y2 - y1)) / 2) * 0.06 / 30


@_check_gene_mutation_rate
@_reset_fitness
@_loop_random_mutations
def mutation_individual_impl(ga, chromosome, _):
    if random.randint(0,1) == 0: # priority 변이
        # Indexes of genes to swap
        index_one = random.randrange(int(len(chromosome) / 2))
        index_two = random.randrange(int(len(chromosome) / 2))

        # Swap genes
        chromosome[index_one], chromosome[index_two] = chromosome[index_two], chromosome[index_one]
    else: # vehicle nums 변이
        index = int(len(chromosome) / 2) + random.randrange(int(len(chromosome) / 2))
        random_value = random.randint(1,M)
        chromosome[index] = random_value

if __name__ == "__main__":
    ga = My_GA()
    ga.pre_setting()
    while ga.active():
        # Evolve only a certain number of generations
        ga.evolve(100)
        #ga.print_population()
        # Print the current generation
        ga.print_generation()
        # Print the best chromosome from that generations population
        ga.print_best_chromosome()
        # If you want to show each population
        # ga.print_population()
        # To divide the print to make it easier to look at
        print('-' * 75)

    ga.graph.lowest_value_chromosome()
    ga.graph.show()
