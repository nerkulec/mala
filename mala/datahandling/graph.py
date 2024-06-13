import os
import numpy as np

from ase import Atoms
from ase.io import read

from tqdm.auto import trange

import dgl
import torch

from functools import lru_cache

from mala.common.parallelizer import printout
from .utils import pickle_cache

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.transformer import get_populated_edge_features


class HashableAtoms(Atoms):
    def __hash__(self):
        return hash(self.get_positions().tobytes())

    def __eq__(self, other):
        return np.allclose(self.get_positions(), other.get_positions())


def get_ldos_positions(cell, nx, ny, nz, corner=False):
    ldos_positions = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    if corner:
        for x in range(nx):
            ldos_positions[x, :, :, 0] = (x) / nx
        for y in range(ny):
            ldos_positions[:, y, :, 1] = (y) / ny
        for z in range(nz):
            ldos_positions[:, :, z, 2] = (z) / nz
    else:
        for x in range(nx):
            ldos_positions[x, :, :, 0] = (x + 0.5) / nx
        for y in range(ny):
            ldos_positions[:, y, :, 1] = (y + 0.5) / ny
        for z in range(nz):
            ldos_positions[:, :, z, 2] = (z + 0.5) / nz

    ldos_positions = ldos_positions.reshape((-1, 3))
    ldos_positions = cell.cartesian_positions(ldos_positions)
    return ldos_positions


def repeat_cell(scaled_positions):
    repeated_positions = [scaled_positions]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue

                repeated_positions.append(
                    scaled_positions + np.array([dx, dy, dz])
                )
    repeated_positions = np.concatenate(repeated_positions)
    return repeated_positions


def prune_positions(all_positions, margin):
    positions = all_positions
    positions = positions[positions[:, 0] > -margin]
    positions = positions[positions[:, 1] > -margin]
    positions = positions[positions[:, 2] > -margin]
    positions = positions[positions[:, 0] < 1 + margin]
    positions = positions[positions[:, 1] < 1 + margin]
    positions = positions[positions[:, 2] < 1 + margin]
    return positions


def assign_relative_pos(graph, coords):
    src, dst = graph.edges()
    graph.edata["rel_pos"] = coords[src] - coords[dst]
    graph.edata["rel_pos"] = graph.edata["rel_pos"].float()
    return graph


def get_periodic_graph(
    graph_repeated_positions, cartesian_positions, n_nodes=256
):
    graph_repeated_positions = assign_relative_pos(
        graph_repeated_positions, cartesian_positions
    )
    src, dst = graph_repeated_positions.edges()
    rel_pos = graph_repeated_positions.edata["rel_pos"]
    src = src % n_nodes
    center_edges = dst < n_nodes
    src = src[center_edges]
    dst = dst[center_edges]
    rel_pos = rel_pos[center_edges]
    periodic_graph = dgl.graph((src, dst), num_nodes=n_nodes)
    periodic_graph.edata["rel_pos"] = rel_pos
    # periodic_graph.ndata['pos'] = cartesian_positions[:n_nodes]
    return periodic_graph


def get_ion_graph(filename, n_closest=8):
    atoms = read(filename)
    atoms.__class__ = HashableAtoms
    cartesian_positions = atoms.get_positions()
    cell = atoms.get_cell()
    scaled_positions = cell.scaled_positions(cartesian_positions)
    repeated_positions = repeat_cell(scaled_positions)
    cartesian_positions = cell.cartesian_positions(repeated_positions)
    cartesian_positions = torch.tensor(
        cartesian_positions, dtype=torch.float32
    )
    graph_repeated_positions = dgl.knn_graph(cartesian_positions, n_closest)
    periodic_graph = get_periodic_graph(
        graph_repeated_positions, cartesian_positions
    )
    return periodic_graph


warned_about_n_batches = False


# TODO: make the graphs uni-bipartite
# @lru_cache(maxsize=1000)
# @pickle_cache(folder_name='ldos_graphs')
def get_ldos_graphs(
    atoms_path,
    ldos,
    ldos_batch_size=1000,
    n_closest_ldos=32,
    ldos_shape=None,
    max_degree=1,
    n_batches=None,
    corner=False,
    randomize_ldos_grid_positions=False,
    seed=None,
):
    atoms = read(atoms_path)
    atoms.__class__ = HashableAtoms
    n_atoms = len(atoms)
    cartesian_ion_positions = atoms.get_positions()
    cell = atoms.get_cell()
    cartesian_ldos_positions = get_ldos_positions(
        cell, *ldos_shape[:-1], corner=corner
    )
    # ! TEMPORARY
    if n_batches is not None:
        global warned_about_n_batches
        if not warned_about_n_batches:
            printout(
                "WARNING: Using 'n_batches' should be only used for development",
                min_verbosity=1,
            )
            warned_about_n_batches = True
        cartesian_ldos_positions = cartesian_ldos_positions[
            : ldos_batch_size * n_batches
        ]

    if randomize_ldos_grid_positions:
        seed = atoms_path + seed
        random_permutation = np.arange(len(cartesian_ldos_positions))
        rs = np.random.RandomState(seed)
        rs.shuffle(random_permutation)
        ldos = ldos[random_permutation]
        cartesian_ldos_positions = cartesian_ldos_positions[random_permutation]

    cartesian_ldos_positions = torch.tensor(
        cartesian_ldos_positions, dtype=torch.float32
    )
    if len(cartesian_ldos_positions) % ldos_batch_size != 0:
        raise ValueError(
            f"{len(cartesian_ldos_positions)=} is not divisible by {ldos_batch_size=}"
        )
    n_ions = len(cartesian_ion_positions)
    scaled_ion_positions = cell.scaled_positions(cartesian_ion_positions)
    repeated_ion_positions = repeat_cell(scaled_ion_positions)
    cartesian_ion_positions = cell.cartesian_positions(repeated_ion_positions)
    cartesian_ion_positions = torch.tensor(
        cartesian_ion_positions, dtype=torch.float32
    )

    ldos_graphs = []
    for i in trange(
        0,
        len(cartesian_ldos_positions),
        ldos_batch_size,
        # leave=False, # test
        desc="Computing LDOS graphs",
    ):
        cartesian_ldos_positions_batch = cartesian_ldos_positions[
            i : i + ldos_batch_size
        ]
        distances = torch.cdist(
            cartesian_ldos_positions_batch, cartesian_ion_positions
        )
        closest_ion_indices = torch.argsort(distances, dim=1)[
            :, :n_closest_ldos
        ]
        dst = torch.arange(ldos_batch_size).repeat_interleave(n_closest_ldos)
        src = closest_ion_indices.flatten()
        rel_pos = (
            cartesian_ldos_positions_batch[dst] - cartesian_ion_positions[src]
        )
        src = src % n_ions
        dst = dst + n_ions
        # ldos_graph = dgl.DGLGraph()
        # ldos_graph.add_nodes(n_ions, ntype='ion')
        # ldos_graph.add_nodes(ldos_batch_size, ntype='grid')
        # ldos_graph.add_edges(src, dst)
        ldos_graph = dgl.graph((src, dst), num_nodes=n_ions + ldos_batch_size)

        ldos_graph.edata["rel_pos"] = rel_pos
        # ldos_graph.ndata['pos'] = torch.cat([cartesian_ion_positions[:n_ions], cartesian_ldos_positions_batch])
        ldos_graphs.append(ldos_graph)

    for j in trange(len(ldos_graphs), desc="Filling LDOS graphs"):
        ldos_batch = torch.tensor(
            ldos[j * ldos_batch_size : (j + 1) * ldos_batch_size],
            dtype=torch.float32,
        )
        ldos_graph = ldos_graphs[j]
        ldos_graph.ndata["target"] = torch.cat(
            [
                torch.zeros((n_atoms, ldos_shape[-1]), dtype=torch.float32),
                ldos_batch,
            ],
            dim=0,
        )
        # ! Assume cell contains only one atoms species
        ldos_graph.ndata["feature"] = torch.cat(
            [
                torch.ones((n_atoms, 1, 1), dtype=torch.float32),
                torch.zeros((ldos_batch_size, 1, 1), dtype=torch.float32),
            ],
            dim=0,
        )
        basis_grid = get_basis(
            ldos_graph.edata["rel_pos"],
            max_degree=max_degree,
            compute_gradients=False,
            use_pad_trick=False,
            amp=torch.is_autocast_enabled(),
        )
        basis_grid = update_basis_with_fused(
            basis_grid,
            max_degree=max_degree,
            use_pad_trick=False,
            fully_fused=True,
        )
        for key in basis_grid.keys():
            ldos_graph.edata["basis_" + key] = basis_grid[key]
        edge_features = get_populated_edge_features(
            ldos_graph.edata["rel_pos"], None
        )
        ldos_graph.edata["edge_features"] = edge_features["0"]

    return ldos_graphs


@lru_cache(maxsize=20)
def load_permuted_ldos(ldos_path, seed, randomize_ldos_grid_positions):
    ldos = np.load(ldos_path)
    ldos = ldos.reshape((-1, ldos.shape[-1]))
    random_permutation = np.arange(len(ldos))
    if randomize_ldos_grid_positions:
        rs = np.random.RandomState(hash(seed) % (2**32))
        rs.shuffle(random_permutation)
    # print(f"load_permuted_ldos {seed=}, {random_permutation[:20]=}")
    ldos = ldos[random_permutation]
    ldos = torch.tensor(ldos, dtype=torch.float32)
    return ldos


def get_ldos_graph_loader(
    atoms_path,
    ldos_path,
    ldos_batch_size=1000,
    n_closest_ldos=32,
    ldos_shape=None,
    corner=False,
    max_degree=1,
    randomize_ldos_grid_positions=False,
    n_samples=None,
    seed=None,
):
    atoms = read(atoms_path)
    atoms.__class__ = HashableAtoms
    n_atoms = len(atoms)
    cartesian_ion_positions = atoms.get_positions()
    cell = atoms.get_cell()
    cartesian_ldos_positions = get_ldos_positions(
        cell, *ldos_shape[:-1], corner=corner
    )

    cartesian_ldos_positions = torch.tensor(
        cartesian_ldos_positions, dtype=torch.float32
    )
    if len(cartesian_ldos_positions) % ldos_batch_size != 0:
        raise ValueError(
            f"len(cartesian_ldos_positions) = {len(cartesian_ldos_positions)} is not divisible by ldos_batch_size = {ldos_batch_size}"
        )
    n_ions = len(cartesian_ion_positions)
    scaled_ion_positions = cell.scaled_positions(cartesian_ion_positions)
    repeated_ion_positions = repeat_cell(scaled_ion_positions)
    cartesian_ion_positions = cell.cartesian_positions(repeated_ion_positions)
    cartesian_ion_positions = torch.tensor(
        cartesian_ion_positions, dtype=torch.float32
    )

    seed = atoms_path + seed
    random_permutation = np.arange(len(cartesian_ldos_positions))
    if randomize_ldos_grid_positions:
        rs = np.random.RandomState(hash(seed) % (2**32))
        rs.shuffle(random_permutation)
    # print(f"get_ldos_graph_loader {seed=}, {random_permutation[:20]=}")
    if n_samples is not None and n_samples < len(cartesian_ldos_positions):
        random_permutation = random_permutation[:n_samples]
    cartesian_ldos_positions = cartesian_ldos_positions[random_permutation]

    def get_ldos_graph(batch_number):
        ldos = load_permuted_ldos(
            ldos_path, seed, randomize_ldos_grid_positions
        )
        cartesian_ldos_positions_batch = cartesian_ldos_positions[
            batch_number
            * ldos_batch_size : (batch_number + 1)
            * ldos_batch_size
        ]
        distances = torch.cdist(
            cartesian_ldos_positions_batch, cartesian_ion_positions
        )
        closest_ion_indices = torch.argsort(distances, dim=1)[
            :, :n_closest_ldos
        ]
        dst = torch.arange(ldos_batch_size).repeat_interleave(n_closest_ldos)
        src = closest_ion_indices.flatten()
        rel_pos = (
            cartesian_ldos_positions_batch[dst] - cartesian_ion_positions[src]
        )
        src = src % n_ions
        dst = dst + n_ions
        ldos_graph = dgl.graph((src, dst), num_nodes=n_ions + ldos_batch_size)
        ldos_graph.edata["rel_pos"] = rel_pos
        # --
        ldos_batch = ldos[
            batch_number
            * ldos_batch_size : (batch_number + 1)
            * ldos_batch_size
        ]
        ldos_graph.ndata["target"] = torch.cat(
            [
                torch.zeros((n_atoms, ldos_shape[-1]), dtype=torch.float32),
                ldos_batch,
            ],
            dim=0,
        )
        # ! Assume cell contains only one atoms species
        ldos_graph.ndata["feature"] = torch.cat(
            [
                torch.ones((n_atoms, 1, 1), dtype=torch.float32),
                torch.zeros((ldos_batch_size, 1, 1), dtype=torch.float32),
            ],
            dim=0,
        )
        basis_grid = get_basis(
            ldos_graph.edata["rel_pos"],
            max_degree=max_degree,
            compute_gradients=False,
            use_pad_trick=False,
            amp=torch.is_autocast_enabled(),
        )
        basis_grid = update_basis_with_fused(
            basis_grid,
            max_degree=max_degree,
            use_pad_trick=False,
            fully_fused=True,
        )
        for key in basis_grid.keys():
            ldos_graph.edata["basis_" + key] = basis_grid[key]
        edge_features = get_populated_edge_features(
            ldos_graph.edata["rel_pos"], None
        )
        ldos_graph.edata["edge_features"] = edge_features["0"]
        # --
        return ldos_graph

    return get_ldos_graph
