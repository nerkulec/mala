from functools import lru_cache
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from mala.common.parallelizer import printout
from .graph import get_ion_graph, get_ldos_graph_loader

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import get_populated_edge_features

from tqdm.auto import tqdm, trange
import time
import concurrent.futures


class Subset:
  def __init__(self, dataset, start_index, end_index):
    self.dataset = dataset
    self.start_index = start_index
    self.end_index = end_index
    
  def __getitem__(self, i):
    if i >= len(self):
      raise IndexError
    return self.dataset[i+self.start_index]
  
  def __len__(self):
    return self.end_index-self.start_index


def expand_ldos_graph(ldos_graph, ldos_batch, ldos_shape, ldos_batch_size, max_degree, n_atoms):
  ldos_graph.ndata['target'] = torch.cat(
    [torch.zeros((n_atoms, ldos_shape[-1]), dtype=torch.float32), ldos_batch], dim=0
  )
  # ! Assume this is a single atom species system
  ldos_graph.ndata['feature'] = torch.cat([
    torch.ones((n_atoms, 1, 1), dtype=torch.float32), torch.zeros((ldos_batch_size, 1, 1), dtype=torch.float32)
  ], dim=0)
  basis_grid = get_basis(
    ldos_graph.edata['rel_pos'], max_degree=max_degree, compute_gradients=False,
    use_pad_trick=False, amp=torch.is_autocast_enabled()
  )
  basis_grid = update_basis_with_fused(
    basis_grid, max_degree=max_degree, use_pad_trick=False, fully_fused=True
  )
  for key in basis_grid.keys():
    ldos_graph.edata['basis_'+key] = basis_grid[key]
  edge_features = get_populated_edge_features(ldos_graph.edata['rel_pos'], None)
  ldos_graph.edata['edge_features'] = edge_features['0']
  return ldos_graph


# @lru_cache(maxsize=2)
def load_ldos(ldos_path):
  ldos = np.load(ldos_path)
  return ldos

# def load_ldos_graph(graph_loader, batch_index):
#   # ldos = load_ldos(ldos_path)
#   # ldos_shape = ldos.shape
#   # ldos = ldos.reshape((-1, ldos_shape[-1]))
#   # ldos = torch.tensor(ldos, dtype=torch.float32)
#   ldos_graph = graph_loader(batch_index)
#   # ldos_graph = expand_ldos_graph(
#   #   ldos_graph, ldos[batch_index*ldos_batch_size:(batch_index+1)*ldos_batch_size],
#   #   ldos_shape, ldos_batch_size, max_degree, n_atoms
#   # )
#   return ldos_graph


class OnTheFlyGraphDataset(Dataset):
  def __init__(
    self, n_closest_ions=8, n_closest_ldos=32, ldos_batch_size=1000,
    max_degree=1, ldos_paths=[], input_paths=[], n_batches=None, n_prefetch=100,
    grid_points_in_corners=False, on_the_fly_shuffling=True, ldos_grid_random_subset=True,
  ):
    super().__init__()
    self.n_snapshots = len(ldos_paths)
    self.ldos_batch_size = ldos_batch_size
    self.n_closest_atoms = n_closest_ions
    self.n_closest_ldos = n_closest_ldos
    self.input_paths = input_paths
    
    self.ldos_paths = ldos_paths
    self.max_degree = max_degree
    self.n_batches = n_batches
    self.n_prefetch = n_prefetch
    self.cache_misses = 0
    self.grid_points_in_corners = grid_points_in_corners
    self.on_the_fly_shuffling = on_the_fly_shuffling
    self.ldos_grid_random_subset = ldos_grid_random_subset
    
    self.pool_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="GraphPrefetcher")
    
    self.ion_graphs = [
      get_ion_graph(input_path, n_closest_ions) for input_path in input_paths
    ]

    for graph_ions in self.ion_graphs:
      basis_ions = get_basis(
        graph_ions.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
        use_pad_trick=False, amp=torch.is_autocast_enabled()
      )
      basis_ions = update_basis_with_fused(
        basis_ions, max_degree=self.max_degree, use_pad_trick=False, fully_fused=True
      )
      for key in basis_ions.keys():
        graph_ions.edata['basis_'+key] = basis_ions[key]

      edge_features = get_populated_edge_features(graph_ions.edata['rel_pos'], None)
      graph_ions.edata['edge_features'] = edge_features['0']


    n_atoms = self.ion_graphs[0].number_of_nodes()
    self.n_atoms = n_atoms
    for i in range(len(self.ion_graphs)):
      self.ion_graphs[i].ndata['feature'] = torch.ones((n_atoms, 1, 1), dtype=torch.float32)

    self.graph_loaders = {}
    self.grid_sizes = []
    for snapshot_index, (input_path, ldos_path) in tqdm(
      enumerate(zip(self.input_paths, self.ldos_paths)),
      total=len(input_paths), desc="Loading LDOS graph loaders"
    ):
      ldos = load_ldos(ldos_path)
      ldos_shape = ldos.shape
      ldos_size = np.prod(ldos_shape[:-1])
      self.n_ldos_batches = ldos_size//self.ldos_batch_size
      self.grid_size = ldos_size
      self.grid_sizes.append(ldos_size)
      self.ldos_dim = ldos_shape[-1]
      ldos = ldos.reshape((-1, ldos_shape[-1]))
      ldos_graph_loader = get_ldos_graph_loader(
        input_path, ldos_path, self.ldos_batch_size, self.n_closest_ldos, ldos_shape,
        corner=self.grid_points_in_corners, randomize_ldos_grid_positions=self.ldos_grid_random_subset,
        seed=input_path, max_degree=self.max_degree
      )
      self.graph_loaders[snapshot_index] = ldos_graph_loader
    self.ldos_graphs_cache = {}
    self.next_cache_index_to_clean = 0
    self.batch_tuples = self.get_batch_tuples()
    self.next_batch_tuples = self.get_batch_tuples()
    
    printout(f"Prefetching {self.n_prefetch} batches")
    start = time.time()
    self.prefetch_mp(self.batch_tuples[:self.n_prefetch])
    end = time.time()
    printout(f"Prefetched {self.n_prefetch} batches in {end-start:.2f}s")
  
  def get_batch_tuples(self):
    batch_tuples = []
    for snapshot_index in range(len(self.input_paths)):
      snapshot_batch_tuples = []
      for batch_index in range(self.n_ldos_batches):
        snapshot_batch_tuples.append((snapshot_index, batch_index))
      if self.on_the_fly_shuffling:
        np.random.shuffle(snapshot_batch_tuples)
      batch_tuples.extend(snapshot_batch_tuples)
    return batch_tuples
    
  def prefetch_mp(self, batch_tuples):
    # ldos_paths = [self.ldos_paths[snapshot_index] for snapshot_index, _ in batch_tuples]
    graph_loaders = [self.graph_loaders[snapshot_index] for snapshot_index, _ in batch_tuples]
    batch_indices = [batch_index for _, batch_index in batch_tuples]
    ldos_graphs = self.pool_executor.map(
      lambda graph_loader, batch_index: graph_loader(batch_index), graph_loaders, batch_indices
    )
    for (snapshot_index, batch_index), ldos_graph in zip(batch_tuples, ldos_graphs):
      self.ldos_graphs_cache[(snapshot_index, batch_index)] = ldos_graph
    
  def clean_cache(self):
    while len(self.ldos_graphs_cache) > self.n_prefetch:
      # snapshot_index = self.next_cache_index_to_clean//self.n_ldos_batches
      # batch_index = self.next_cache_index_to_clean%self.n_ldos_batches
      snapshot_index, batch_index = self.batch_tuples[self.next_cache_index_to_clean]
      try:
        del self.ldos_graphs_cache[(snapshot_index, batch_index)]
      except KeyError:
        pass
      self.next_cache_index_to_clean += 1
      self.next_cache_index_to_clean %= self.n_snapshots*self.n_ldos_batches
    
  def get_filled_ldos_graph(self, i):
    snapshot_index, batch_index = self.batch_tuples[i]
    if (snapshot_index, batch_index) in self.ldos_graphs_cache:
      return self.ldos_graphs_cache[(snapshot_index, batch_index)]
    # printout(f"Warning: cache miss for ({i}/{self.n_snapshots*self.n_ldos_batches})")
    self.cache_misses += 1
    # ldos_path = self.ldos_paths[snapshot_index]
    graph_loader = self.graph_loaders[snapshot_index]
    ldos_graph = graph_loader(batch_index)
    return ldos_graph
  
  def save_ldos_graph_and_cleanup(self, snapshot_index, batch_index, ldos_graph):
    self.ldos_graphs_cache[(snapshot_index, batch_index)] = ldos_graph
    self.clean_cache()

  def __getitem__(self, i):
    # snapshot_index = i//self.n_ldos_batches
    # batch_index = i%self.n_ldos_batches
    snapshot_index, batch_index = self.batch_tuples[i]
    ion_graph = self.ion_graphs[snapshot_index]
    ldos_graph = self.get_filled_ldos_graph(i)
    
    prefetch_index = i + self.n_prefetch
    prefetch_for_next_epoch = False
    if prefetch_index >= len(self.batch_tuples):
      prefetch_for_next_epoch = True
    prefetch_index %= len(self.batch_tuples)
    
    # remove later
    assert len(self.batch_tuples) == self.n_snapshots*self.n_ldos_batches
    # prefetch_snapshot_index = prefetch_index//self.n_ldos_batches
    # prefetch_batch_index = prefetch_index%self.n_ldos_batches
    if not prefetch_for_next_epoch:
      prefetch_snapshot_index, prefetch_batch_index = self.batch_tuples[prefetch_index]
    else:
      prefetch_snapshot_index, prefetch_batch_index = self.next_batch_tuples[prefetch_index]
    
    future_graph = self.pool_executor.submit(
      lambda graph_loader, batch_index: graph_loader(batch_index), self.graph_loaders[prefetch_snapshot_index], prefetch_batch_index,
    )
    future_graph.add_done_callback(lambda ldos_graph:
      self.save_ldos_graph_and_cleanup(prefetch_snapshot_index, prefetch_batch_index, ldos_graph.result())
    )
    
    if i == len(self.batch_tuples)-1:
      self.batch_tuples = self.next_batch_tuples
      self.next_batch_tuples = self.get_batch_tuples()
      printout(f"Cache misses: {self.cache_misses}/{len(self.batch_tuples)}")
      self.cache_misses = 0
    
    return ion_graph, ldos_graph

  def __len__(self):
    return self.n_snapshots*self.n_ldos_batches

