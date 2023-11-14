import numpy as np

# operacja A**-1 podnosi kazda liczbe w array do potegi -1 tzn znajduje odwrotnosc danej liczby
# czyli nie jest to odwrotnosc amcierzy (w takim sensie algebraicznym)
A = np.array([[1, 2], [3, 4]], dtype=float)
A_prim = A**-1

print("Odwrotność wersja array:")
print(A_prim)

# operacja B**-1 dokonuje odwrócenia macierzy (w sensie matematycznym)
B = np.matrix([[1, 2], [3, 4]])
B_prim = B**-1

print("Odwrotność wersja matrix:")
print(B_prim)
