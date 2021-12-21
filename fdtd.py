import numpy as np
from itertools import product
from tqdm import tqdm

DIM = 3
X, Y, Z = tuple(range(DIM))
VACUUM_PERMITTIVITY = 8.85e-12  # 真空の誘電率
VACUUM_PERMEABILITY = 1.26e-6   # 真空の透磁率
LIGHT_SPEED = 299_792_458       # 光速

# 電場の回転
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

# 磁場の回転
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

# 静電場計算
def calculate_static_E(q_func, eps, E_coords):
  E = np.zeros_like(E_coords)
  q_dist = q_func(E_coords[..., X], E_coords[..., Y], E_coords[..., Z])
  for idx, _ in tqdm(np.ndenumerate(E[..., X])):
    # 各点pの電場を求める
    p = E_coords[idx]
    r = p - E_coords
    e = q_dist[..., np.newaxis] / (4 * np.pi * eps) * r / (np.linalg.norm(r) ** 3)
    e = np.sum(e, axis=tuple(range(e.ndim - 1)))
    E[idx] = e
  return E

# CollocatedグリッドとYeeグリッドの相互変換
def convert_E_coll_and_yee(x, y, z):
  return np.concatenate((
    (x[:-1, :-1, :-1, np.newaxis] + x[1:, :-1, :-1, np.newaxis]) / 2,
    (y[:-1, :-1, :-1, np.newaxis] + y[:-1, 1:, :-1, np.newaxis]) / 2,
    (z[:-1, :-1, :-1, np.newaxis] + z[:-1, :-1, 1:, np.newaxis]) / 2,
  ), axis=-1)

def convert_B_coll_and_yee(x, y, z):
  return np.concatenate((
    (x[:-1, :-1, :-1, np.newaxis] + x[:-1, 1:, 1:, np.newaxis]) / 2,
    (y[:-1, :-1, :-1, np.newaxis] + y[1:, :-1, 1:, np.newaxis]) / 2,
    (z[:-1, :-1, :-1, np.newaxis] + z[1:, 1:, :-1, np.newaxis]) / 2,
  ), axis=-1)

def convert_J_coll_and_yee(x, y, z):
  return np.concatenate((
    (x[:-1, :-1, :-1, np.newaxis] + x[1:, :-1, :-1, np.newaxis]) / 2,
    (y[:-1, :-1, :-1, np.newaxis] + y[:-1, 1:, :-1, np.newaxis]) / 2,
    (z[:-1, :-1, :-1, np.newaxis] + z[:-1, :-1, 1:, np.newaxis]) / 2,
  ), axis=-1)


class Grid():
  def __init__(self,
    x_slice, y_slice, z_slice,
    relative_permittivity = 1.0, relative_permeavility = 1.0 # 比誘電率・比透磁率
  ):
    assert all(isinstance(s, slice) for s in [x_slice, y_slice, z_slice])
    # 計算範囲
    self.x_slice = x_slice  # x_slice.startからstopまで計算
    self.y_slice = y_slice
    self.z_slice = z_slice

    # 離散化幅
    self.dx = x_slice.step
    self.dy = y_slice.step
    self.dz = z_slice.step

    # Collocated座標（通常の座標）
    x = np.arange(x_slice.start - x_slice.step, x_slice.stop + 2 * x_slice.step, x_slice.step)  # 計算用に少し幅を持たせて計算する
    y = np.arange(y_slice.start - y_slice.step, y_slice.stop + 2 * y_slice.step, y_slice.step)
    z = np.arange(z_slice.start - z_slice.step, z_slice.stop + 2 * z_slice.step, z_slice.step)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    self.calc_coords = np.concatenate((x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]), axis=-1)

    # Yee座標
    self.E_yee_coords = convert_E_coll_and_yee(self.calc_coords[..., X], self.calc_coords[..., Y], self.calc_coords[..., Z]) # 電場のYeeグリッドの各ベクトルの座標
    self.B_yee_coords = convert_B_coll_and_yee(self.calc_coords[..., X], self.calc_coords[..., Y], self.calc_coords[..., Z]) # 磁場
    self.J_yee_coords = convert_J_coll_and_yee(self.calc_coords[..., X], self.calc_coords[..., Y], self.calc_coords[..., Z]) # 電流密度

    # 有効な値を計算できる座標
    self.coords = self.calc_coords[1:-1, 1:-1, 1:-1]

    assert self.E_yee_coords.shape == self.B_yee_coords.shape == self.J_yee_coords.shape
    sizes = self.E_yee_coords.shape[:-1]

    # 境界上のインデックス（境界条件計算用）
    self.border_idx = [[], [], []]
    for idx_tuple in product(*(range(size) for size in sizes)):
      if any(idx == 0 or idx == size - 1 for idx, size in zip(idx_tuple, sizes)):
        for axis in range(DIM):
          self.border_idx[axis].append(idx_tuple[axis])
    self.border_idx = tuple(self.border_idx)
    
    # 境界上の各要素に対して1つ内側の要素のインデックス（境界条件計算用）
    self.inner_idx = [[], [], []]
    for idx_tuple in zip(*self.border_idx):
      idx_tuple = np.array(idx_tuple)
      idx_tuple += (idx_tuple == 0).astype(int) - (idx_tuple == np.array(sizes) - 1).astype(int)
      for axis in range(DIM):
        self.inner_idx[axis].append(idx_tuple[axis])
    self.inner_idx = tuple(self.inner_idx)

    # 誘電率、透磁率
    self.eps = VACUUM_PERMITTIVITY * relative_permittivity
    self.mu = VACUUM_PERMEABILITY * relative_permeavility 
 
  def initialize(self, q_func, J_func): # q_func(x, y, z), J_func(x, y, z)
    self.old_E = calculate_static_E(q_func, self.eps, self.E_yee_coords)
    self.old_B = np.zeros_like(self.B_yee_coords)
    self.old_J = J_func(self.J_yee_coords[..., X], self.J_yee_coords[..., Y], self.J_yee_coords[..., Z])

  def update(self, dt, J_func):
    self.J_yee = J_func(self.J_yee_coords[..., X], self.J_yee_coords[..., Y], self.J_yee_coords[..., Z])
    curl = curl_E(self.old_E, self.dx, self.dy, self.dz)
    self.B_yee = self.old_B - dt * curl
    curl = curl_B(self.B_yee, self.dx, self.dy, self.dz)
    self.E_yee = self.old_E + dt / (self.eps * self.mu) * curl - dt / self.eps * self.J_yee

  def update_border(self, dt):
    dis = np.linalg.norm(self.E_yee_coords[self.border_idx] - self.E_yee_coords[self.inner_idx])
    self.E_yee[self.border_idx] = self.old_E[self.inner_idx] + \
      (dt * LIGHT_SPEED - dis) / (dt * LIGHT_SPEED + dis) * (self.E_yee[self.inner_idx] - self.old_E[self.border_idx])
    dis = np.linalg.norm(self.B_yee_coords[self.border_idx] - self.B_yee_coords[self.inner_idx])
    self.B_yee[self.border_idx] = self.old_B[self.inner_idx] + \
      (dt * LIGHT_SPEED - dis) / (dt * LIGHT_SPEED + dis) * (self.B_yee[self.inner_idx] - self.old_B[self.border_idx])
  
  def calc_coll(self):
    E = self.old_E
    B = (self.B_yee + self.old_B) / 2
    J = (self.J_yee + self.old_J) / 2
    self.E_coll = convert_E_coll_and_yee(E[..., X], E[..., Y], E[..., Z])
    self.B_coll = convert_B_coll_and_yee(B[..., X], B[..., Y], B[..., Z])
    self.J_coll = convert_J_coll_and_yee(J[..., X], J[..., Y], J[..., Z])
    self.old_E = self.E_yee
    self.old_B = self.B_yee
    self.old_J = self.J_yee


class Calculator():
  def __init__(self, grid, dt, t_stop, q_func, q_vel_func):
    self.grid = grid
    self.dt = dt
    assert all(LIGHT_SPEED * dt <= d / np.sqrt(DIM) for d in [self.grid.dx, self.grid.dy, self.grid.dz])
    self.t_size = int(t_stop / dt)
    self.q_func = lambda x, y, z: q_func(0, x, y, z)
    self.J_func = lambda x, y, z: q_func(self.t_idx * dt + dt / 2, x, y, z)[..., np.newaxis] * q_vel_func(self.t_idx * dt + dt / 2, x, y, z)
    self.__iter__()
    
  def __len__(self):
    return self.t_size
  
  def __iter__(self):
    self.t_idx = 0
    self.grid.initialize(self.q_func, self.J_func)
    return self

  def __next__(self):
    if self.t_idx >= self.t_size:
      raise StopIteration()
    self.grid.update(self.dt, self.J_func)
    self.grid.update_border(self.dt)
    self.grid.calc_coll()
    self.t_idx += 1
    return self.grid.coords, self.grid.E_coll, self.grid.B_coll, self.grid.J_coll
