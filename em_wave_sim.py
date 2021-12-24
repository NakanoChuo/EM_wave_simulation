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
t_max = 1 / freq * 5

# 電気双極子の電流密度
def dipole_J_func(t, x, y, z):
  # +電荷の分布
  pos = np.array([0, 0, amp * np.sin(2 * np.pi * freq * t)])  # 分布の中心位置
  p_dist = np.exp(-((x - pos[X]) ** 2) / 0.01) * np.exp(-((y - pos[Y]) ** 2) / 0.01) * np.exp(-((z - pos[Z]) ** 2) / 0.01)
  # -電荷の分布
  n_dist = np.exp(-((-x - pos[X]) ** 2) / 0.01) * np.exp(-((-y - pos[Y]) ** 2) / 0.01) * np.exp(-((-z - pos[Z]) ** 2) / 0.01)
  # 電荷の速度
  q_vel = np.array([0, 0, 2 * np.pi * freq * amp * np.cos(2 * np.pi * freq * t)])

  return q * p_dist[..., np.newaxis] * q_vel + (-q) * n_dist[..., np.newaxis] * -q_vel

calculator = Calculator(
  Grid(slice(x_min, x_max, dx), slice(y_min, y_max, dy), slice(z_min, z_max, dz)),
  dt, t_max, dipole_J_func
)

import matplotlib.pyplot as plt
import matplotlib.animation as anime

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.view_init(elev=20, azim=-10)

with tqdm(total=len(calculator)) as progress_bar:
  def init():
    pass

  def plot(frame_num):
    coords, E, B, J = next(calculator)
    x, y, z = coords[..., X], coords[..., Y], coords[..., Z]
    x_size, _, z_size, _ = coords.shape

    B *= LIGHT_SPEED
    J *= 10

    ax1.clear()
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax1.set_xticks([x_min, x_max]); ax1.set_yticks([y_min, y_max]); ax1.set_zticks([0])
    ax1.plot([x_min, x_max, x_max], [y_min, y_min, y_max], [0, 0, 0], linewidth=1, color='darkgray')
    ax1.quiver(x[::2, ::2, z_size // 2], y[::2, ::2, z_size // 2], z[::2, ::2, z_size // 2], E[::2, ::2, z_size // 2, X], E[::2, ::2, z_size // 2, Y], E[::2, ::2, z_size // 2, Z], color='red', linewidth=0.5)
    ax1.quiver(x[::2, ::2, z_size // 2], y[::2, ::2, z_size // 2], z[::2, ::2, z_size // 2], B[::2, ::2, z_size // 2, X], B[::2, ::2, z_size // 2, Y], B[::2, ::2, z_size // 2, Z], color='green', linewidth=0.5)
    ax1.quiver(x[::2, ::2, z_size // 2], y[::2, ::2, z_size // 2], z[::2, ::2, z_size // 2], J[::2, ::2, z_size // 2, X], J[::2, ::2, z_size // 2, Y], J[::2, ::2, z_size // 2, Z], color='blue', linewidth=0.5)

    ax2.clear()
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
    ax2.set_xticks([0]); ax2.set_yticks([y_min, y_max]); ax2.set_zticks([z_min, z_max])
    ax2.plot([0, 0, 0], [y_min, y_min, y_max], [z_min, z_max, z_max], linewidth=1, color='darkgray')
    ax2.quiver(x[x_size // 2, ::2, ::2], y[x_size // 2, ::2, ::2], z[x_size // 2, ::2, ::2], E[x_size // 2, ::2, ::2, X], E[x_size // 2, ::2, ::2, Y], E[x_size // 2, ::2, ::2, Z], color='red', linewidth=0.5, label='E')
    ax2.quiver(x[x_size // 2, ::2, ::2], y[x_size // 2, ::2, ::2], z[x_size // 2, ::2, ::2], B[x_size // 2, ::2, ::2, X], B[x_size // 2, ::2, ::2, Y], B[x_size // 2, ::2, ::2, Z], color='green', linewidth=0.5, label='B')
    ax2.quiver(x[x_size // 2, ::2, ::2], y[x_size // 2, ::2, ::2], z[x_size // 2, ::2, ::2], J[x_size // 2, ::2, ::2, X], J[x_size // 2, ::2, ::2, Y], J[x_size // 2, ::2, ::2, Z], color='blue', linewidth=0.5, label='J')
    ax2.legend(loc=(0.8, -0.3))

    progress_bar.update()
    
  animation = anime.FuncAnimation(fig, plot, interval=1, frames=len(calculator), init_func=init)
  animation.save('output.gif', writer='imagemagick')
