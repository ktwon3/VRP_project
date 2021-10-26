import numpy as np
import pickle
import matplotlib.pyplot as plt

class Map:
    def __init__(self, N, max_size, mode):
        """
        :param N: 수요지 수
        :param max_size: 맵의 최대 크기 (-max_size ~ max_size까지 맵 설정됨)
        """
        if N >= 1 and isinstance(N, int):
            self.map = np.zeros((N+1, 2))
            self.N = N
            self.max = max_size
            self.mode = mode
            self.center = None
        else: raise Exception("N error")

    def create_map(self):
        mode = self.mode
        if mode == "R":
            temp = np.random.rand(2 * self.N)
            temp = temp.reshape((self.N, 2))
            temp -= 0.5
            temp *= 2 * self.max
            self.map[1:] = temp
        elif mode == "C":
            if self.N % 5 == 0:
                boundary_num = int(self.N / 5)
                center = np.random.uniform(-1 * self.max * 0.65, self.max * 0.65, 2 * boundary_num).reshape(boundary_num,2)
                map_ = np.zeros((self.N, 2))
                for i in range(len(center)):
                    center_x, center_y = center[i]
                    print('x,y =', center_x, center_y)
                    boundary = np.random.normal(0, 1, 2 * boundary_num).reshape(boundary_num, 2)
                    boundary *= 100
                    map_[i * 5:i * 5 + 5, :] = boundary + center[i]
                self.map[1:] = map_
                self.center = center
        else: raise Exception("No valid mode")

    def save_map(self):
        with open("data/data_first.pickle", "wb") as fw:
            pickle.dump(self.map, fw)

    def map_plot(self):
        x_list, y_list =[x for x,y in self.map], [y for x,y in self.map]
        plt.title('Map')
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.xlim((-1000,1000))
        plt.ylim((-1000, 1000))
        plt.scatter(0,0)
        plt.scatter(x_list[1:], y_list[1:])
        if self.mode == "C":
            x_list, y_list = [x for x, y in self.center], [y for x, y in self.center]
            plt.scatter(x_list, y_list)

        plt.show()

    def print_map(self):
        print(self.map)


if __name__ == "__main__":
    m = Map(5, 1000, "R")
    m.create_map()
    m.print_map()
    m.map_plot()
    if input("save? : ") == "Y":
        m.save_map()
