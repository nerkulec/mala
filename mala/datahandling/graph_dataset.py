import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .graph import get_ion_graph, get_ldos_graphs

class GraphDataset(Dataset):
  def __init__(
    self, n_closest_ions=8, n_closest_ldos=32, ldos_batch_size=1000,
    ldos_paths=[], input_paths=[]
  ):
    super().__init__()
    self.n_snapshots = len(ldos_paths)
    self.ldos_batch_size = ldos_batch_size
    self.ion_graphs = [
      get_ion_graph(input_path, n_closest_ions) for input_path in input_paths
    ]
    n_atoms = self.ion_graphs[0].number_of_nodes()
    self.n_atoms = n_atoms
    for i in range(len(self.ion_graphs)):
      self.ion_graphs[i].ndata['feature'] = torch.ones((n_atoms, 1, 1), dtype=torch.float32)

    self.ldos_graphs = [
      get_ldos_graphs(input_path, ldos_batch_size, n_closest_ldos) for input_path in input_paths
    ]

    self.grid_sizes = []
    self.n_ldos_batches = None

    for list_i, ldos_path in enumerate(ldos_paths):
      ldos = np.load(ldos_path)
      ldos_shape = ldos.shape

      ldos_size = np.prod(ldos_shape[:-1])
      self.grid_size = ldos_size
      if self.n_ldos_batches is not None and self.n_ldos_batches != ldos_size//ldos_batch_size:
        raise Exception("ldos batch sizes are not consistent")
      self.n_ldos_batches = ldos_size//ldos_batch_size
      self.grid_sizes.append(ldos_size)

      self.ldos_dim = ldos_shape[-1]
      ldos = ldos.reshape((-1, ldos_shape[-1]))

      for j in range(self.n_ldos_batches):
        ldos_batch = torch.tensor(
          ldos[j*ldos_batch_size:(j+1)*ldos_batch_size], dtype=torch.float32
        )
        self.ldos_graphs[list_i][j].ndata['target'] = torch.cat(
          [torch.zeros((n_atoms, ldos_shape[-1]), dtype=torch.float32), ldos_batch], dim=0
        )
        # ! Assume atom is hydrogen
        self.ldos_graphs[list_i][j].ndata['feature'] = torch.cat([
          torch.ones((n_atoms, 1, 1), dtype=torch.float32), torch.zeros((ldos_batch_size, 1, 1), dtype=torch.float32)
        ], dim=0)

  def __getitem__(self, i):
    snapshot_index = i//self.n_ldos_batches
    ldos_index = i%self.n_ldos_batches
    ion_graph = self.ion_graphs[snapshot_index]
    ldos_graph = self.ldos_graphs[snapshot_index][ldos_index]
    return ion_graph, ldos_graph

  def __len__(self):
    return self.n_snapshots*self.n_ldos_batches
  
