import matplotlib.pyplot as plt
import numpy as np

# numer indeksu: 473560
# start,stop,liczba pkt do wygenerowania
x = np.linspace(-6, 6, 1000)
y = (5 - 4) * x**2 + (6 - 4) * x + (0 - 6)
g = np.exp(x) / (np.exp(x) + 1)

plt.plot(x, y, color="blue")
plt.plot(x, g, color="red")
plt.xlabel("Oś x")
plt.ylabel("Oś y")
plt.title("Zadanie 2")
plt.grid(True)
plt.show()
