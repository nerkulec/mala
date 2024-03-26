import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .graph import get_ion_graph, get_ldos_graphs

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import get_populated_edge_features
from mala.datahandling.utils import sanitize, pickle_cache
from functools import lru_cache
from mala.common.parallelizer import printout
from tqdm.auto import tqdm, trange
from ase.io import read



@lru_cache(maxsize=1)
@pickle_cache(folder_name='ldos_graphs')
def load_ldos_graphs(ldos_path, input_path, ldos_batch_size, n_closest_ldos, max_degree, n_batches, corner):
  ldos = np.load(ldos_path)
  ldos_shape = ldos.shape
  ldos_graphs = get_ldos_graphs(
    input_path, ldos_batch_size, n_closest_ldos,
    n_batches=n_batches, ldos_shape=ldos_shape, corner=corner
  )
  
  atoms = read(input_path)
  cartesian_ion_positions = atoms.get_positions()
  n_atoms = len(cartesian_ion_positions)

  ldos = ldos.reshape((-1, ldos_shape[-1]))
  for j in trange(len(ldos_graphs), desc="Filling LDOS graphs"):
    ldos_batch = torch.tensor(
      ldos[j*ldos_batch_size:(j+1)*ldos_batch_size], dtype=torch.float32
    )
    ldos_graph = ldos_graphs[j]
    ldos_graph.ndata['target'] = torch.cat(
      [torch.zeros((n_atoms, ldos_shape[-1]), dtype=torch.float32), ldos_batch], dim=0
    )
    # ! Assume only one type of atoms
    ldos_graph.ndata['feature'] = torch.cat([
      torch.ones((n_atoms, 1, 1), dtype=torch.float32),
      torch.zeros((ldos_batch_size, 1, 1), dtype=torch.float32)
    ], dim=0)
    basis_grid = get_basis(
      ldos_graph.edata['rel_pos'], max_degree=max_degree, compute_gradients=False, # Max degree already present here
      use_pad_trick=False, amp=torch.is_autocast_enabled()
    )
    basis_grid = update_basis_with_fused(
      basis_grid, max_degree=max_degree, use_pad_trick=False, fully_fused=True # Max degree already present here
    )
    for key in basis_grid.keys():
      ldos_graph.edata['basis_'+key] = basis_grid[key]
    edge_features = get_populated_edge_features(ldos_graph.edata['rel_pos'], None)
    ldos_graph.edata['edge_features'] = edge_features['0']
  return ldos_graphs
  

class LazyGraphDataset(Dataset):
  def __init__(
    self, n_closest_ions=8, n_closest_ldos=32, ldos_batch_size=1000,
    max_degree=1, ldos_paths=[], input_paths=[], n_batches=None
  ):
    super().__init__()
    self.n_snapshots = len(ldos_paths)
    self.ldos_batch_size = ldos_batch_size
    self.max_degree = max_degree
    self.n_closest_ions = n_closest_ions
    self.n_closest_ldos = n_closest_ldos
    self.ldos_paths = ldos_paths
    self.input_paths = input_paths
    self.n_batches = n_batches
    self.ion_graphs = [
      get_ion_graph(input_path, self.n_closest_ions) for input_path in input_paths
    ]

    for graph_ions in self.ion_graphs:
      basis_ions = get_basis(
        graph_ions.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False, # ! Max degree already present here
        use_pad_trick=False, amp=torch.is_autocast_enabled()
      )
      basis_ions = update_basis_with_fused(
        basis_ions, max_degree=self.max_degree, use_pad_trick=False, fully_fused=True # ! Max degree already present here
      )
      for key in basis_ions.keys():
        graph_ions.edata['basis_'+key] = basis_ions[key]

      edge_features = get_populated_edge_features(graph_ions.edata['rel_pos'], None)
      graph_ions.edata['edge_features'] = edge_features['0']

    # ! Check memory here
    
    self.n_atoms = self.ion_graphs[0].number_of_nodes()
    for i in range(len(self.ion_graphs)):
      # Assumed atom is hydrogen
      self.ion_graphs[i].ndata['feature'] = torch.ones((self.n_atoms, 1, 1), dtype=torch.float32)

    # If there is no "ldos_graphs" folder, create it
    if not os.path.exists("ldos_graphs"):
      os.makedirs("ldos_graphs")

    printout("Initial LDOS graph generation", min_verbosity=2)
    dataset_length = 0
    for i, ldos_path in tqdm(
      enumerate(ldos_paths), total=len(ldos_paths),
      desc="Lazy loading LDOS graphs"
    ):
      ldos_graphs = self._get_ldos_graphs(i)
      self.n_ldos_batches = len(ldos_graphs)
      dataset_length += len(ldos_graphs)
    
    self.num_batches = dataset_length
      
  def _get_ldos_graphs(self, i):
    ldos_path = self.ldos_paths[i]
    input_path = self.input_paths[i]
    ldos_graphs = load_ldos_graphs(
      ldos_path, input_path, self.ldos_batch_size, self.n_closest_ldos,
      self.max_degree, self.n_batches, corner=self.params.grid_points_in_corners
    )
    self.ldos_dim = ldos_graphs[0].ndata['target'].shape[-1]
    return ldos_graphs
        

  def __getitem__(self, i):
    snapshot_index = i//self.n_ldos_batches
    ldos_index = i%self.n_ldos_batches
    ion_graph = self.ion_graphs[snapshot_index]
    ldos_graphs = self._get_ldos_graphs(snapshot_index)
    ldos_graph = ldos_graphs[ldos_index]
    return ion_graph, ldos_graph

  def __len__(self):
    return self.num_batches
  
