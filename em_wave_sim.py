from fdtd import Grid, Calculator
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
x_min = -0.5
y_min = -0.5
z_min = -0.5
x_max = 0.5
y_max = 0.5
z_max = 0.5
t_max = 1 / freq * 1

# 電荷分布
def q_func(t, x, y, z):
  q_center_pos = np.array([0, amp * np.sin(2 * np.pi * freq * t), 0])
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with tqdm(total=len(calculator)) as progress_bar:
  def init():
    pass

  #for frame_num in range(len(calculator)):
  def plot(frame_num):
    coords, E, B, J = next(calculator)
    x, y, z = coords[..., X], coords[..., Y], coords[..., Z]

    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.quiver(x[::3, ::3, ::3], y[::3, ::3, ::3], z[::3, ::3, ::3], E[::3, ::3, ::3, X], E[::3, ::3, ::3, Y], E[::3, ::3, ::3, Z], color='C2')

    progress_bar.update()
    
  animation = anime.FuncAnimation(fig, plot, interval=1, frames=len(calculator), init_func=init)
  animation.save('output.gif', writer='imagemagick')
