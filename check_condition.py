import numpy as np
from necessary_parameter import *

def check_one_visit(ga):
    for i in range(1, N+1):
        j = i
        if np.sum(ga.x[i,:,:]) != 1: return False
        if np.sum(ga.x[:, j, :]) != 1: return False
    return True

def check_continuous(ga):
    for p in range(1,N+1):
        for m in range(1,M+1):
            if np.sum(ga.x[:,p,m]) - np.sum(ga.x[p,:,m]) != 0: return False
    return True

def check_volume(ga):
    for m in range(1, M+1):
        count = 0
        for i in range(1, N+1):  # for문 대신 행렬 내적으로 처리 가능해보임 차피 D[0] = 0이니 생략
            count += np.sum(ga.x[i, :, m]) * ga.D[i]
        if count > C: return False
    return True

def check_garage(ga):
    for m in range(1, M+1):
        if np.sum(ga.x[0,:,m]) > 1 or np.sum(ga.x[:,0,m]) > 1: return False
    return True

def check_nomove(ga):
    for m in range(1, M+1):
        if np.any(np.diag(ga.x[:,:,m]) != 0):
            return False
    return True
