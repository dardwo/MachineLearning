import numpy as np

# w zadaniu pomijam fakt, ze wyznacznik macierzy X raz X transponowany wynosi 0
# wykonanie działań z pomocą array
X = np.array([[1, 2, 3], [1, 3, 6]])
y = np.array([[5], [6]])

XT = X.transpose()

print(np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), y))

# sprawdzenie 2 sposobem; wykonanie wszystkich działań na matrix
B = np.matrix([[1, 2, 3], [1, 3, 6]])
y = np.matrix([[5], [6]])

BT = B.transpose()
wynik = np.dot(BT, B)
print(np.dot(np.dot(wynik**-1, BT), y))
