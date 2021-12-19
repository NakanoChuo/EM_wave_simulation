import numpy as np
from itertools import product
from tqdm import tqdm

DIM = 3
VACUUM_PERMITTIVITY = 8.85e-12  # 真空の誘電率
VACUUM_PERMEABILITY = 1.26e-6   # 真空の透磁率
LIGHT_SPEED = 299_792_458       # 光速

relative_permititvity = 1.0     # 比誘電率
relative_permeavility = 1.0     # 比透磁率
eps = VACUUM_PERMITTIVITY * relative_permititvity # 誘電率
mu = VACUUM_PERMEABILITY * relative_permeavility  # 透磁率

freq = 1 / 2.7e-9 # 周波数[Hz]
amp = 0.1         # 振幅[m]
q = 1e-10         # 電荷[C]

# 離散化幅
deltas = [0.04, 0.04, 0.04] # 空間
dt = 7e-11                  # 時間
assert all(LIGHT_SPEED * dt <= d / np.sqrt(DIM) for d in deltas)  # Courant安定条件？

# 計算範囲
x_min = -0.5
y_min = -0.5
z_min = -0.5
min_vals = [x_min, y_min, z_min]
x_max = 0.5
y_max = 0.5
z_max = 0.5
max_vals = [x_max, y_max, z_max]

t_max = 1 / freq * 10

# 離散化数
sizes = [int((max_v - min_v) / d) for max_v, min_v, d in zip(max_vals, min_vals, deltas)]
t_size = int(t_max / dt)

# 座標
x = np.arange(x_min, x_max, deltas[0])
y = np.arange(y_min, y_max, deltas[1])
z = np.arange(z_min, z_max, deltas[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
# 位置ベクトル
coords = np.concatenate((X[..., np.newaxis], Y[..., np.newaxis], Z[..., np.newaxis]), axis=-1)

# 普通のグリッド（表示用）
e_coll = np.zeros([t_size] + sizes + [DIM])
b_coll = np.zeros([t_size] + sizes + [DIM])
j_coll = np.zeros([t_size] + sizes + [DIM])
# 上3つの配列の有効な値が計算できる範囲
valid_t_indices = range(1, t_size - 1)
valid_pos_indices = (slice(1, -1),) * DIM
valid_indices = (valid_t_indices,) + valid_pos_indices

# Yeeグリッド（計算用）
e_yee = np.zeros([t_size - 1] + [size - 1 for size in sizes] + [DIM])
b_yee = np.zeros([t_size - 1] + [size - 1 for size in sizes] + [DIM])
j_yee = np.zeros([t_size - 1] + [size - 1 for size in sizes] + [DIM])

from scipy.signal import fftconvolve
def interconvert_e_yee_and_coll(e_array): # e_yee と e_coll の相互変換
  convs = []
  for axis in range(DIM):
    kernel_shape = (1,) + (1,) * axis + (2,) + (1,) * (DIM - axis - 1) + (1,)
    crop_indices = (slice(None),) + tuple(slice(0, s - 1) for s in e_array.shape[1:1+DIM]) + (axis, np.newaxis)
    convs.append(fftconvolve(e_array, np.ones(kernel_shape) / 2, mode='valid')[crop_indices])
  return np.concatenate(convs, axis=-1)

def interconvert_b_yee_and_coll(b_array):
  convs = []
  for axis in range(DIM):
    kernel_shape = (2,) + (2,) * axis + (1,) + (2,) * (DIM - axis - 1) + (1,)
    crop_indices = (slice(None),) + tuple(slice(0, s - 1) for s in b_array.shape[1:1+DIM]) + (axis, np.newaxis)
    convs.append(fftconvolve(b_array, np.ones(kernel_shape) / 8, mode='valid')[crop_indices])
  return np.concatenate(convs, axis=-1)

def interconvert_j_yee_and_coll(j_array):
  convs = []
  for axis in range(DIM):
    kernel_shape = (2,) + (1,) * axis + (2,) + (1,) * (DIM - axis - 1) + (1,)
    crop_indices = (slice(None),) + tuple(slice(0, s - 1) for s in j_array.shape[1:1+DIM]) + (axis, np.newaxis)
    convs.append(fftconvolve(j_array, np.ones(kernel_shape) / 4, mode='valid')[crop_indices])
  return np.concatenate(convs, axis=-1)

# 電荷分布
def get_q_dist(t, X, Y, Z):
  q_center_pos = np.array([0, amp * np.sin(2 * np.pi * freq * t), 0])
  return q * np.exp(-((X - q_center_pos[0]) ** 2) / 0.01) * np.exp(-((Y - q_center_pos[1]) ** 2) / 0.01) * np.exp(-((Z - q_center_pos[2]) ** 2) / 0.01)

# 電荷の速度
def get_q_vel_vec(t):
  return np.array([0, 0, 2 * np.pi * freq * amp * np.cos(2 * np.pi * freq * t)])

# 電流密度
for t_id in range(t_size):
  q_dist = get_q_dist(t_id * dt, X, Y, Z)
  q_vel = get_q_vel_vec(t_id * dt)
  j_coll[t_id] = q_dist[..., np.newaxis] * q_vel
j_yee = interconvert_j_yee_and_coll(j_coll)

# 初期値
q_dist = get_q_dist(0, X, Y, Z)
for id_tuple in tqdm(tuple(product(*(range(size) for size in sizes)))):
  # 各点pの電場を求める
  p = coords[id_tuple]
  r = p - coords
  e_coords = q_dist[..., np.newaxis] / (4 * np.pi * eps) * r / (np.linalg.norm(r) ** 3) # 各電荷による点pの電場を求め
  e = np.sum(e_coords, axis=tuple(range(e_coords.ndim - 1)))                            # 合計する
  e_coll[0][id_tuple] = e
e_yee[0] = interconvert_e_yee_and_coll(e_coll[0, np.newaxis])
# q_dist = np.exp(-((X) ** 2) / 0.02) * np.exp(-((Y) ** 2) / 0.02) * np.exp(-((Z) ** 2) / 0.02)
# e_coll[0] = q_dist[..., np.newaxis] * np.array([0,0,0.5])
# e_yee[0] = interconvert_e_yee_and_coll(e_coll[0, np.newaxis])

dx, dy, dz = deltas

def curl_E(E, dx, dy, dz):
  assert E.ndim == DIM + 1 and E.shape[-1] == DIM
  curl = np.zeros_like(E)
  curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2]) / dy
  curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1]) / dz
  curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0]) / dz
  curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2]) / dx
  curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1]) / dx
  curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0]) / dy
  return curl

def curl_B(B, dx, dy, dz):
  assert B.ndim == DIM + 1 and B.shape[-1] == DIM
  curl = np.zeros_like(B)
  curl[:, 1:, :, 0] += (B[:, 1:, :, 2] - B[:, :-1, :, 2]) / dy
  curl[:, :, 1:, 0] -= (B[:, :, 1:, 1] - B[:, :, :-1, 1]) / dz
  curl[:, :, 1:, 1] += (B[:, :, 1:, 0] - B[:, :, :-1, 0]) / dz
  curl[1:, :, :, 1] -= (B[1:, :, :, 2] - B[:-1, :, :, 2]) / dx
  curl[1:, :, :, 2] += (B[1:, :, :, 1] - B[:-1, :, :, 1]) / dx
  curl[:, 1:, :, 2] -= (B[:, 1:, :, 0] - B[:, :-1, :, 0]) / dy
  return curl

# 更新
for t_id in tqdm(tuple(range(0, t_size - 1))):
  if t_id > 0:
    e_yee[t_id] = e_yee[t_id - 1] + dt / (eps * mu) * curl_B(b_yee[t_id - 1], *deltas) - dt / eps * j_yee[t_id - 1]

  b_yee[t_id] = (b_yee[t_id - 1] if t_id > 0 else 0) - dt * curl_E(e_yee[t_id], *deltas)

  # 境界条件（Murの吸収境界）
  for id_tuple in product(*(range(size - 1) for size in sizes)):
    if any(_id == 0 or _id == size - 2 for _id, size in zip(id_tuple, sizes)):
      next_id_tuple = tuple(_id + (_id == 0) - (_id == size - 2) for _id, size in zip(id_tuple, sizes))
      dis = np.linalg.norm(coords[id_tuple] - coords[next_id_tuple])
      e_yee[t_id][id_tuple] = e_yee[t_id - 1][next_id_tuple] + (dt * LIGHT_SPEED - dis) / (dt * LIGHT_SPEED + dis) * (e_yee[t_id][next_id_tuple] - e_yee[t_id - 1][id_tuple])
      b_yee[t_id][id_tuple] = b_yee[t_id - 1][next_id_tuple] + (dt * LIGHT_SPEED - dis) / (dt * LIGHT_SPEED + dis) * (b_yee[t_id][next_id_tuple] - b_yee[t_id - 1][id_tuple])

e_coll[valid_indices] = interconvert_e_yee_and_coll(e_yee[1:])
b_coll[valid_indices] = interconvert_b_yee_and_coll(b_yee)

import matplotlib.pyplot as plt
import matplotlib.animation as anime

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with tqdm(total=len(valid_t_indices)) as progress_bar:
  #for frame_num in range(len(valid_t_indices)):
  def plot(frame_num):
    progress_bar.update()
    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.quiver(X[::3, ::3, ::3], Y[::3, ::3, ::3], Z[::3, ::3, ::3], e_coll[frame_num, ::3, ::3, ::3, 0], e_coll[frame_num, ::3, ::3, ::3, 1], e_coll[frame_num, ::3, ::3, ::3, 2], color='C2')
    #ax.quiver(X[:-1:3, :-1:3, :-1:3], Y[:-1:3, :-1:3, :-1:3], Z[:-1:3, :-1:3, :-1:3], e_yee[frame_num, ::3, ::3, ::3, 0], e_yee[frame_num, ::3, ::3, ::3, 1], e_yee[frame_num, ::3, ::3, ::3, 2], color='C2')
    #ax.quiver(X[:-1:3, :-1:3, :-1:3], Y[:-1:3, :-1:3, :-1:3], Z[:-1:3, :-1:3, :-1:3], b_yee[frame_num, ::3, ::3, ::3, 0], b_yee[frame_num, ::3, ::3, ::3, 1], b_yee[frame_num, ::3, ::3, ::3, 2], color='C3')
    #ax.quiver(X[:-1:3, :-1:3, :-1:3], Y[:-1:3, :-1:3, :-1:3], Z[:-1:3, :-1:3, :-1:3], j_yee[frame_num, ::3, ::3, ::3, 0], j_yee[frame_num, ::3, ::3, ::3, 1], j_yee[frame_num, ::3, ::3, ::3, 2], color='C1')
  animation = anime.FuncAnimation(fig, plot, interval=1, frames=len(valid_t_indices))
  animation.save('output.gif', writer='imagemagick')
