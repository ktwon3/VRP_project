from EasyGA import *
from necessary_parameter import *
from check_condition import *
from necessary_parameter import *


def get_fitness(chromosome):
    """
    fitness_function
    :param chromosome:
    :return: fitness(float), 논문의 fitness 가져옴
    """
    temp_ga = My_GA()
    temp_ga.set_x(chromosome)
    temp_ga.set_s(s)
    temp_ga.setMap(map_)
    temp_ga.set_demend(D)
    temp_ga.setTime(mode=time_mode)

    if not temp_ga.check_condition(): return float('inf')

    fitness = 0
    for m in range(1, M+1):
        fitness += np.sum(temp_ga.time_arr * temp_ga.x[:, :, m])
    fitness += np.sum(temp_ga.s)

    return fitness

class My_GA(GA):
    def setMap(self, m):
        """
        Map 지정
        """
        self.Map = m

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

def get_all_priority(nums):
    result = []
    nums.sort()
    for i in range(len(nums)):
        num = nums[i]
        before_result = get_all_priority(nums[:i] + nums[i+1:])
        if len(before_result) == 0: result.append([num])
        else:
            for br in before_result:
                result.append([num] + br)
    return result

b = get_all_priority(list(range(1, N+1)))
print(b)
chromosome_list = []

for b1 in b:
    for i1 in range(1,M+1):
        for i2 in range(1,M+1):
            for i3 in range(1,M+1):
                for i4 in range(1,M+1):
                    for i5 in range(1, M+1):
                        chromosome_list.append(b1 + [i1,i2,i3,i4,i5])

best = {float("inf"): []}
for chromosome in chromosome_list:
    fitness = get_fitness(chromosome)
    if list(best.keys())[0] > fitness:
        best = {fitness: chromosome}
print(best)
print(get_fitness([3,2,4,5,1,3,2,1,1,2]))
