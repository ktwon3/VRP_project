import pickle
import matplotlib.pyplot as plt
import numpy as np
SAVE_FILE = "result/Rtype_method4.pickle"
with open(SAVE_FILE, "rb") as fr:
    data = pickle.load(fr)

print(data[0])
print(data[1])
plt.plot(range(1,501), data[2])
plt.xlabel('benchmark')
plt.ylabel('cost (min)')
plt.title('<method 3 - C type>  benchmark - cost graph')
plt.show()