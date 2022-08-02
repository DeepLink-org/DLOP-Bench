import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("conv2d_top3.csv")
ids = data["id"]
networks = data["source_neural_network"].values
running_time = [float(i)*1000 for i in data["running_time"].values]
# print(networks)
# print(networks[0], networks[1], networks[2])
x = range(len(running_time))

# x = range(3)
# print(x)
plt.bar(x, running_time)
plt.xlabel("parameter configs")
plt.ylabel("running time(ms)")
plt.savefig("./conv2d_top3.png")
plt.title("Performance of Conv2d")
# plt.xticks(x, ids)

plt.show()



