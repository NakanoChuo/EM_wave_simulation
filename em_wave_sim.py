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

fig1 = plt.figure(dpi=200)
ax_E = fig1.add_subplot(221, projection='3d')
ax_B = fig1.add_subplot(222, projection='3d')
ax_J = fig1.add_subplot(223, projection='3d')

with tqdm(total=len(calculator)) as progress_bar:
  def init():
    pass

  def plot(frame_num):
    coords, E, B, J = next(calculator)
    x, y, z = coords[..., X], coords[..., Y], coords[..., Z]

    B *= LIGHT_SPEED
    J *= 10

    ax_E.clear(); ax_E.set_xlim(x_min, x_max); ax_E.set_ylim(y_min, y_max); ax_E.set_zlim(z_min, z_max)
    ax_E.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], E[::3, ::3, ::3, X], E[::3, ::3, ::3, Y], E[::3, ::3, ::3, Z], color='C2')
    ax_B.clear(); ax_B.set_xlim(x_min, x_max); ax_B.set_ylim(y_min, y_max); ax_B.set_zlim(z_min, z_max)
    ax_B.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], B[::3, ::3, ::3, X], B[::3, ::3, ::3, Y], B[::3, ::3, ::3, Z], color='C1')
    ax_J.clear(); ax_J.set_xlim(x_min, x_max); ax_J.set_ylim(y_min, y_max); ax_J.set_zlim(z_min, z_max)
    ax_J.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], J[::3, ::3, ::3, X], J[::3, ::3, ::3, Y], J[::3, ::3, ::3, Z], color='C3')

    progress_bar.update()
    
  animation = anime.FuncAnimation(fig1, plot, interval=1, frames=len(calculator), init_func=init)
  animation.save('output.gif', writer='imagemagick')
