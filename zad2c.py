import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = -(X**2 + Y**3)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="inferno")

ax.set_xlabel("oś x")
ax.set_ylabel("oś y")
ax.set_zlabel("oś z")

plt.show()
