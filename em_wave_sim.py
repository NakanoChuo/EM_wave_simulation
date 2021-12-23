from fdtd import Grid, Calculator, LIGHT_SPEED
import numpy as np
from tqdm import tqdm

DIM = 3
X, Y, Z = tuple(range(DIM))

freq = 1 / 2.7e-9 # 周波数[Hz]
amp = 0.1         # 振幅[m]
q = 1e-10         # 電荷[C]

# 離散化幅
dx = 0.04
dy = 0.04
dz = 0.04
dt = 7e-11

# 計算範囲
x_min = -0.6
y_min = -0.6
z_min = -0.6
x_max = 0.6
y_max = 0.6
z_max = 0.6
t_max = 1 / freq * 2

# 電荷分布
def q_func(t, x, y, z):
  q_center_pos = np.array([0, 0, amp * np.sin(2 * np.pi * freq * t)])
  return q * np.exp(-((x - q_center_pos[X]) ** 2) / 0.01) * np.exp(-((y - q_center_pos[Y]) ** 2) / 0.01) * np.exp(-((z - q_center_pos[Z]) ** 2) / 0.01)

# 電荷の速度
def q_vel_func(t, x, y, z):
  return np.array([0, 0, 2 * np.pi * freq * amp * np.cos(2 * np.pi * freq * t)])

calculator = Calculator(
  Grid(slice(x_min, x_max, dx), slice(y_min, y_max, dy), slice(z_min, z_max, dz)),
  dt, t_max, q_func, q_vel_func
)

import matplotlib.pyplot as plt
import matplotlib.animation as anime

fig = plt.figure(dpi=200)
ax_E = fig.add_subplot(3, 4, 1, projection='3d')
ax_B = fig.add_subplot(3, 4, 2, projection='3d')
ax_J = fig.add_subplot(3, 4, 3, projection='3d')
ax1 = fig.add_subplot(3, 4, (5, 10), projection='3d')
ax2 = fig.add_subplot(3, 4, (7, 12), projection='3d')

with tqdm(total=len(calculator)) as progress_bar:
  def init():
    pass

  def plot(frame_num):
    coords, E, B, J = next(calculator)
    x, y, z = coords[..., X], coords[..., Y], coords[..., Z]
    x_size, _, z_size, _ = coords.shape

    B *= LIGHT_SPEED
    J *= 10

    ax_E.clear(); ax_E.set_xlim(x_min, x_max); ax_E.set_ylim(y_min, y_max); ax_E.set_zlim(z_min, z_max)
    ax_E.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], E[::3, ::3, ::3, X], E[::3, ::3, ::3, Y], E[::3, ::3, ::3, Z], color='C2')
    ax_B.clear(); ax_B.set_xlim(x_min, x_max); ax_B.set_ylim(y_min, y_max); ax_B.set_zlim(z_min, z_max)
    ax_B.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], B[::3, ::3, ::3, X], B[::3, ::3, ::3, Y], B[::3, ::3, ::3, Z], color='C1')
    ax_J.clear(); ax_J.set_xlim(x_min, x_max); ax_J.set_ylim(y_min, y_max); ax_J.set_zlim(z_min, z_max)
    ax_J.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], J[::3, ::3, ::3, X], J[::3, ::3, ::3, Y], J[::3, ::3, ::3, Z], color='C3')

    ax1.clear(); ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax1.quiver(x[::3, ::3, z_size // 2], y[::3, ::3, z_size // 2], z[::3, ::3, z_size // 2], E[::3, ::3, z_size // 2, X], E[::3, ::3, z_size // 2, Y], E[::3, ::3, z_size // 2, Z], color='C2')
    ax1.quiver(x[::3, ::3, z_size // 2], y[::3, ::3, z_size // 2], z[::3, ::3, z_size // 2], B[::3, ::3, z_size // 2, X], B[::3, ::3, z_size // 2, Y], B[::3, ::3, z_size // 2, Z], color='C1')
    ax1.quiver(x[::3, ::3, z_size // 2], y[::3, ::3, z_size // 2], z[::3, ::3, z_size // 2], J[::3, ::3, z_size // 2, X], J[::3, ::3, z_size // 2, Y], J[::3, ::3, z_size // 2, Z], color='C3')

    ax2.clear(); ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
    ax2.quiver(x[x_size // 2, ::3, ::3], y[x_size // 2, ::3, ::3], z[x_size // 2, ::3, ::3], E[x_size // 2, ::3, ::3, X], E[x_size // 2, ::3, ::3, Y], E[x_size // 2, ::3, ::3, Z], color='C2')
    ax2.quiver(x[x_size // 2, ::3, ::3], y[x_size // 2, ::3, ::3], z[x_size // 2, ::3, ::3], B[x_size // 2, ::3, ::3, X], B[x_size // 2, ::3, ::3, Y], B[x_size // 2, ::3, ::3, Z], color='C1')
    ax2.quiver(x[x_size // 2, ::3, ::3], y[x_size // 2, ::3, ::3], z[x_size // 2, ::3, ::3], J[x_size // 2, ::3, ::3, X], J[x_size // 2, ::3, ::3, Y], J[x_size // 2, ::3, ::3, Z], color='C3')

    progress_bar.update()
    
  animation = anime.FuncAnimation(fig, plot, interval=1, frames=len(calculator), init_func=init)
  animation.save('output.gif', writer='imagemagick')
