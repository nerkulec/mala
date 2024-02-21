import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .graph import get_ion_graph, get_ldos_graphs

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import get_populated_edge_features

from tqdm.auto import tqdm, trange

class GraphDataset(Dataset):
  def __init__(
    self, n_closest_ions=8, n_closest_ldos=32, ldos_batch_size=1000,
    ldos_paths=[], input_paths=[], n_batches=None
  ):
    super().__init__()
    self.n_snapshots = len(ldos_paths)
    self.ldos_batch_size = ldos_batch_size
    self.ion_graphs = [
      get_ion_graph(input_path, n_closest_ions) for input_path in input_paths
    ]

    for graph_ions in self.ion_graphs:
      basis_ions = get_basis(
        graph_ions.edata['rel_pos'], max_degree=1, compute_gradients=False,
        use_pad_trick=False, amp=torch.is_autocast_enabled()
      )
      basis_ions = update_basis_with_fused(
        basis_ions, max_degree=1, use_pad_trick=False, fully_fused=True
      )
      for key in basis_ions.keys():
        graph_ions.edata['basis_'+key] = basis_ions[key]

      edge_features = get_populated_edge_features(graph_ions.edata['rel_pos'], None)
      graph_ions.edata['edge_features'] = edge_features['0']


    n_atoms = self.ion_graphs[0].number_of_nodes()
    self.n_atoms = n_atoms
    for i in range(len(self.ion_graphs)):
      self.ion_graphs[i].ndata['feature'] = torch.ones((n_atoms, 1, 1), dtype=torch.float32)

    self.ldos_graphs = []
    self.grid_sizes = []

    for list_i, (ldos_path, input_path) in enumerate(tqdm(zip(ldos_paths, input_paths))):
      ldos = np.load(ldos_path)
      ldos_shape = ldos.shape

      self.ldos_graphs.append(
        get_ldos_graphs(
          input_path, ldos_batch_size, n_closest_ldos,
          n_batches=n_batches, ldos_shape=ldos_shape
        )
      )

      ldos_size = np.prod(ldos_shape[:-1])
      self.grid_size = ldos_size
      self.grid_sizes.append(ldos_size)

      self.ldos_dim = ldos_shape[-1]
      ldos = ldos.reshape((-1, ldos_shape[-1]))
      self.n_ldos_batches = len(self.ldos_graphs[0])
      for j in trange(len(self.ldos_graphs[list_i])):
        ldos_batch = torch.tensor(
          ldos[j*ldos_batch_size:(j+1)*ldos_batch_size], dtype=torch.float32
        )
        ldos_graph = self.ldos_graphs[list_i][j]
        ldos_graph.ndata['target'] = torch.cat(
          [torch.zeros((n_atoms, ldos_shape[-1]), dtype=torch.float32), ldos_batch], dim=0
        )
        # ! Assume atom is hydrogen
        ldos_graph.ndata['feature'] = torch.cat([
          torch.ones((n_atoms, 1, 1), dtype=torch.float32), torch.zeros((ldos_batch_size, 1, 1), dtype=torch.float32)
        ], dim=0)
        basis_grid = get_basis(
          ldos_graph.edata['rel_pos'], max_degree=1, compute_gradients=False,
          use_pad_trick=False, amp=torch.is_autocast_enabled()
        )
        basis_grid = update_basis_with_fused(
          basis_grid, max_degree=1, use_pad_trick=False, fully_fused=True
        )
        for key in basis_grid.keys():
          ldos_graph.edata['basis_'+key] = basis_grid[key]
        edge_features = get_populated_edge_features(ldos_graph.edata['rel_pos'], None)
        ldos_graph.edata['edge_features'] = edge_features['0']
        

  def __getitem__(self, i):
    snapshot_index = i//self.n_ldos_batches
    ldos_index = i%self.n_ldos_batches
    ion_graph = self.ion_graphs[snapshot_index]
    ldos_graph = self.ldos_graphs[snapshot_index][ldos_index]
    return ion_graph, ldos_graph

  def __len__(self):
    return sum(len(ldos_graph) for ldos_graph in self.ldos_graphs)
  
