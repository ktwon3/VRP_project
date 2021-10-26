import numpy as np
import pickle

def load_map(map_name):
    with open(map_name, "rb") as fr:
        data = pickle.load(fr)
    return data

"""
Rtype 방법
map_name = "data/data_Rtype.pickle"
N = 25  # 수요지 수, 차고지 제외
M = 4  # 차량 최대 수
C = 40  # 차량 최대 적재량
s = [0] + [1] * N  # 서비스 시간
D = np.array([0] + [5] * N) # 수요량
map_ = load_map(map_name)  # map, 인덱스 0은 차고지
time_mode = 3  # setTime에 쓰이는 변수
"""


"""
C-type 방법
"""
map_name = "data/data_Ctype.pickle"
N = 25  # 수요지 수, 차고지 제외
M = 5  # 차량 최대 수
C = 30  # 차량 최대 적재량
s = [0] + [1] * N  # 서비스 시간
D = np.array([0] + [5] * N) # 수요량
map_ = load_map(map_name)  # map, 인덱스 0은 차고지
time_mode = 3  # setTime에 쓰이는 변수

"""
검증 실험
map_name = "data/data_first.pickle"
N = 5  # 수요지 수, 차고지 제외
M = 3  # 차량 최대 수
C = 10  # 차량 최대 적재량
s = [0] + [1] * N  # 서비스 시간
D = np.array([0] + [5] * N) # 수요량
map_ = load_map(map_name)  # map, 인덱스 0은 차고지
#map_ = np.array([[0,0], [-3, -3], [-2,2], [1, 1], [2,-2], [4,0]])  # map, 인덱스 0은 차고지
time_mode = 2  # setTime에 쓰이는 변수
"""
