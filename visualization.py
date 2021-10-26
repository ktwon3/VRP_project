import pickle
import matplotlib.pyplot as plt
import numpy as np
SAVE_FILE = "result/Ctype_method3.pickle"
with open(SAVE_FILE, "rb") as fr:
    data = pickle.load(fr)

plt.plot(range(1, 502, 10), data[2])
plt.xlabel('benchmark')
plt.ylabel('cost (min)')
plt.title('<method 3 - C type>  benchmark - cost graph')
plt.show()