import os
import numpy as np

import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read

from tqdm import trange

import dgl
import torch

from functools import lru_cache


class HashableAtoms(Atoms):
  def __hash__(self):
    return hash(self.get_positions().tobytes())

  def __eq__(self, other):
    return np.allclose(self.get_positions(), other.get_positions())


def get_ldos_positions(cell, nx=90, ny=90, nz=60):
  ldos_positions = np.zeros((nx, ny, nz, 3), dtype=np.float32)
  # I assume ldos values are evaluated at the center of each voxel
  for x in range(nx):
    ldos_positions[x, :, :, 0] = (x+0.5)/nx
  for y in range(ny):
    ldos_positions[:, y, :, 1] = (y+0.5)/ny
  for z in range(nz):
    ldos_positions[:, :, z, 2] = (z+0.5)/nz

  ldos_positions = ldos_positions.reshape((-1, 3))
  ldos_positions = cell.cartesian_positions(ldos_positions)
  return ldos_positions


def get_ldos(
  ldos_path = "/bigdata/casus/wdm/Bartek_H2/H128/ldos/",
  num_snapshots=20, nx=90, ny=90, nz=60, ldos_dim=201
):
  ldos = np.zeros((num_snapshots, nx, ny, nz, ldos_dim), dtype=np.float32)
  
  for i in trange(num_snapshots):
    ldos[i] = np.load(os.path.join(ldos_path, f'H_snapshot{i}.out.npy'))
  ldos = ldos.reshape((num_snapshots, -1, ldos_dim))
  return ldos


def show_positions(scaled_positions):
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.scatter(
    scaled_positions[:, 0], scaled_positions[:, 1],
    s=40/scaled_positions[:, 2]**0.5, c=scaled_positions[:, 2], cmap='hot'
  )
  plt.subplot(1, 3, 2)
  plt.scatter(
    scaled_positions[:, 0], scaled_positions[:, 2],
    s=40/scaled_positions[:, 1]**0.5, c=scaled_positions[:, 1], cmap='hot'
  )
  plt.subplot(1, 3, 3)
  plt.scatter(
    scaled_positions[:, 1], scaled_positions[:, 2],
    s=40/scaled_positions[:, 0]**0.5, c=scaled_positions[:, 0], cmap='hot'
  )


def repeat_cell(scaled_positions):
  repeated_positions = [scaled_positions]
  for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
      for dz in [-1, 0, 1]:
        if dx == dy == dz == 0:
          continue
        
        repeated_positions.append(scaled_positions + np.array([dx, dy, dz]))
  repeated_positions = np.concatenate(repeated_positions)
  return repeated_positions


def prune_positions(all_positions, margin):
  positions = all_positions
  positions = positions[positions[:, 0] > -margin]
  positions = positions[positions[:, 1] > -margin]
  positions = positions[positions[:, 2] > -margin]
  positions = positions[positions[:, 0] < 1+margin]
  positions = positions[positions[:, 1] < 1+margin]
  positions = positions[positions[:, 2] < 1+margin]
  return positions


def assign_relative_pos(graph, coords):
  src, dst = graph.edges()
  graph.edata['rel_pos'] = coords[src] - coords[dst]
  graph.edata['rel_pos'] = graph.edata['rel_pos'].float()
  return graph


def get_periodic_graph(graph_repeated_positions, cartesian_positions, n_nodes=128):
  graph_repeated_positions = assign_relative_pos(graph_repeated_positions, cartesian_positions)
  src, dst = graph_repeated_positions.edges()
  rel_pos = graph_repeated_positions.edata['rel_pos']
  src = src % n_nodes
  center_edges = dst < n_nodes
  src = src[center_edges]
  dst = dst[center_edges]
  rel_pos = rel_pos[center_edges]
  periodic_graph = dgl.graph((src, dst), num_nodes=n_nodes)
  periodic_graph.edata['rel_pos'] = rel_pos
  periodic_graph.ndata['pos'] = cartesian_positions[:n_nodes]
  return periodic_graph


def get_ion_graph(filename, n_closest=8):
  atoms = read(filename)
  atoms.__class__ = HashableAtoms
  cartesian_positions = atoms.get_positions()
  cell = atoms.get_cell()
  scaled_positions = cell.scaled_positions(cartesian_positions)
  repeated_positions = repeat_cell(scaled_positions)
  cartesian_positions = cell.cartesian_positions(repeated_positions)
  cartesian_positions = torch.tensor(cartesian_positions, dtype=torch.float32)
  graph_repeated_positions = dgl.knn_graph(cartesian_positions, n_closest)
  periodic_graph = get_periodic_graph(graph_repeated_positions, cartesian_positions)
  return periodic_graph


@lru_cache(maxsize=1000)
def get_ldos_graphs(filename, ldos_batch_size=1000, n_closest_ldos=32):
  atoms = read(filename)
  atoms.__class__ = HashableAtoms
  cartesian_ion_positions = atoms.get_positions()
  cell = atoms.get_cell()
  cartesian_ldos_positions = get_ldos_positions(cell)
  cartesian_ldos_positions = torch.tensor(cartesian_ldos_positions, dtype=torch.float32)
  n_ions = len(cartesian_ion_positions)
  scaled_ion_positions = cell.scaled_positions(cartesian_ion_positions)
  repeated_ion_positions = repeat_cell(scaled_ion_positions)
  cartesian_ion_positions = cell.cartesian_positions(repeated_ion_positions)
  cartesian_ion_positions = torch.tensor(cartesian_ion_positions, dtype=torch.float32)
  ldos_graphs = []
  for i in trange(0, len(cartesian_ldos_positions), ldos_batch_size):
    cartesian_ldos_positions_batch = cartesian_ldos_positions[i:i+ldos_batch_size]
    distances = torch.cdist(cartesian_ldos_positions_batch, cartesian_ion_positions)
    closest_ion_indices = torch.argsort(distances, dim=1)[:, :n_closest_ldos]
    dst = torch.arange(ldos_batch_size).repeat_interleave(n_closest_ldos)
    src = closest_ion_indices.flatten()
    rel_pos = cartesian_ldos_positions_batch[dst] - cartesian_ion_positions[src]
    src = src % n_ions
    dst = dst + n_ions
    ldos_graph = dgl.graph((src, dst), num_nodes=n_ions+ldos_batch_size)
    ldos_graph.edata['rel_pos'] = rel_pos
    ldos_graph.ndata['pos'] = torch.cat([cartesian_ion_positions[:n_ions], cartesian_ldos_positions_batch])
    ldos_graphs.append(ldos_graph)
  return ldos_graphs


