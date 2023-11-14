import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def h(t, X):
    return t[0] + t[1] * X


def lin_reg(t, X, y, alpha, epsilon):
    m = len(y)

    cost_history = []

    for _ in range(m):
        dif = h(t, X) - y
        t[0] -= alpha * (1 / m) * np.sum(dif)
        t[1] -= alpha * (1 / m) * np.sum(dif * X)
        cost = (1 / (2 * m)) * np.sum(dif**2)

        if (
            len(cost_history) > 1
            and abs(cost_history[-1] - cost_history[-2]) <= epsilon
        ):
            break

        cost_history.append(cost)

    return t, cost_history


data = pd.read_csv("fires_thefts.csv", header=None)
fires = data.iloc[:, 0]
thefts = data.iloc[:, 1]
t = np.zeros(2)

alpha = 0.0001
epsilon = 0.0001

t, cost_history = lin_reg(t, fires, thefts, alpha, epsilon)

print(f"t_0 = {t[0]}")
print(f"t_1 = {t[1]}\n")

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Steps")
plt.ylabel("Cost")
plt.show()

fires_pred = [50, 100, 200]

for fires_count in fires_pred:
    pred_thefts = h(t, fires_count)
    print(f"{fires_count} fires: {pred_thefts:.2f} thefts")

alphas = [0.0001, 0.01, 0.1]  # Lista różnych alpha
epsilons = [0.00001, 0.0001, 0.001]  # Lista różnych epsilonow


plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    t, cost_history = lin_reg(t, fires, thefts, alpha, epsilon)
    plt.plot(range(len(cost_history)), cost_history, label=f"epsilon = {epsilon}")

plt.xlabel("Steps")
plt.ylabel("Cost")
plt.title("Zależność kosztu od różnych epsilonów")

plt.legend()
plt.show()
