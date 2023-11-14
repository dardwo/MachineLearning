import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data2.csv")

k1 = data.iloc[:,2]
k2 = data.iloc[:,3]

plt.scatter(k1,k2)
plt.xlabel("Dane z kolumny 2")
plt.ylabel("Dane z kolumny 3")
plt.title("Wykres")

plt.show()