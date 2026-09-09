"""Normalize dataset source formats into one common Graph format.

This file deliberately stops after loading.  It does not split nodes, build an
inductive graph, sample negatives, or write DEAL artifacts.

Supported source families:
    * a saved DEAL folder: A_sp.npz, X_sp.npz, z.npy
    * an Amazon benchmark .npz file
    * a SNAP edge-list file, such as facebook_combined.txt.gz
    * PPI's graph.json + feature/graph-id files (one Graph per component)
"""

from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass
class Graph:
    """One graph in the format shared by the rest of the learning pipeline."""

    adjacency: sp.csr_matrix
    features: sp.csr_matrix | None
    labels: np.ndarray | None
    edges: np.ndarray
    directed: bool

    def __post_init__(self):
        self.adjacency = self.adjacency.tocsr()

        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        if self.features is not None:
            self.features = self.features.tocsr()
            if self.features.shape[0] != self.num_nodes:
                raise ValueError("feature rows must match adjacency rows")
        if self.labels is not None and self.labels.shape[0] != self.num_nodes:
            raise ValueError("labels must have one row per node")
        if self.edges.ndim != 2 or self.edges.shape[0] != 2:
            raise ValueError("edges must have shape [2, number of edges]")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= self.num_nodes):
            raise ValueError("edge endpoint is outside the graph")
        if not self.directed and (self.adjacency != self.adjacency.T).nnz:
            raise ValueError("an undirected graph must have symmetric adjacency")

    @property
    def num_nodes(self):
        return self.adjacency.shape[0]

    @property
    def num_edges(self):
        return self.edges.shape[1]


def edges_from_adjacency(adjacency):
    """Return every stored adjacency entry as an edge array shaped [2, E]."""
    source, target = adjacency.nonzero()
    return np.vstack((source, target)).astype(np.int64)


def binary_adjacency(edges, num_nodes, directed, keep_self_loops=False):
    """Build a binary sparse adjacency while applying the graph's direction rule."""
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must have shape [2, number of edges]")

    source = edges[0]
    target = edges[1]
    if not keep_self_loops:
        not_self_loop = source != target
        source = source[not_self_loop]
        target = target[not_self_loop]

    if not directed:
        source, target = (
            np.concatenate((source, target)),
            np.concatenate((target, source)),
        )

    adjacency = sp.csr_matrix(
        (np.ones(source.size, dtype=np.float32), (source, target)),
        shape=(num_nodes, num_nodes),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def load_deal_folder(folder, directed):
    """Load a complete DEAL source folder already stored as sparse artifacts."""
    folder = Path(folder)
    adjacency = sp.load_npz(folder / "A_sp.npz").tocsr()
    features = sp.load_npz(folder / "X_sp.npz").tocsr()
    labels = np.load(folder / "z.npy")
    return Graph(adjacency, features, labels, edges_from_adjacency(adjacency), directed)


def csr_from_npz_contents(data, prefix):
    """Rebuild one CSR matrix stored as prefix_data/indices/indptr/shape."""
    return sp.csr_matrix(
        (data[f"{prefix}_data"], data[f"{prefix}_indices"], data[f"{prefix}_indptr"]),
        shape=data[f"{prefix}_shape"],
    )


def load_amazon_npz(path):
    """Load Amazon's raw benchmark .npz into an undirected feature-bearing Graph."""
    with np.load(path) as data:
        raw_adjacency = csr_from_npz_contents(data, "adj")
        features = csr_from_npz_contents(data, "attr").astype(np.float32)
        labels = data["labels"].astype(np.int64)

    # Co-purchase is undirected even though the raw adjacency can be asymmetric.
    raw_adjacency.setdiag(0)
    raw_adjacency.eliminate_zeros()
    adjacency = (raw_adjacency + raw_adjacency.T).astype(bool).astype(np.float32).tocsr()
    return Graph(adjacency, features, labels, edges_from_adjacency(adjacency), directed=False)


def open_text(path):
    """Open either a plain-text or gzipped edge-list source file."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_snap_edge_list(path, directed):
    """Load SNAP's ``source target`` edge list and remap its ids to 0..N-1.

    SNAP topology files have no node features or labels.  Those fields remain
    None until a feature-specific loader supplies them.
    """
    with open_text(path) as handle:
        raw_pairs = np.loadtxt(handle, dtype=np.int64).reshape(-1, 2)

    raw_node_ids = np.unique(raw_pairs.reshape(-1))
    edges = np.vstack((
        np.searchsorted(raw_node_ids, raw_pairs[:, 0]),
        np.searchsorted(raw_node_ids, raw_pairs[:, 1]),
    ))
    adjacency = binary_adjacency(
        edges,
        num_nodes=len(raw_node_ids),
        directed=directed,
        keep_self_loops=directed,
    )
    return Graph(adjacency, None, None, edges_from_adjacency(adjacency), directed)


def load_ppi_graphs(folder):
    """Load PPI's disjoint source graphs as separately remapped Graph objects.

    PPI labels are multi-label rather than one class per node.  They are kept
    as a two-dimensional array; each graph uses local ids.
    """
    folder = Path(folder)
    loaded_graphs = []

    for split in ("train", "valid", "test"):
        features = np.load(folder / f"{split}_feats.npy").astype(np.float32)
        labels = np.load(folder / f"{split}_labels.npy").astype(np.int8)
        graph_ids = np.load(folder / f"{split}_graph_id.npy")
        graph_ids = graph_ids - graph_ids.min()
        with (folder / f"{split}_graph.json").open(encoding="utf-8") as handle:
            links = json.load(handle)["links"]

        graph_count = int(graph_ids.max()) + 1
        edges_by_graph = [[] for _ in range(graph_count)]
        for link in links:
            source = int(link["source"])
            target = int(link["target"])
            graph_id = int(graph_ids[source])
            if source != target and graph_id == int(graph_ids[target]):
                edges_by_graph[graph_id].append((source, target))

        for graph_id, global_edges in enumerate(edges_by_graph):
            members = np.flatnonzero(graph_ids == graph_id)
            local_id = {int(full_id): local for local, full_id in enumerate(members)}
            if global_edges:
                edges = np.array(
                    [[local_id[source] for source, _ in global_edges],
                     [local_id[target] for _, target in global_edges]],
                    dtype=np.int64,
                )
            else:
                edges = np.empty((2, 0), dtype=np.int64)

            adjacency = binary_adjacency(edges, len(members), directed=False)
            graph = Graph(
                adjacency,
                sp.csr_matrix(features[members]),
                labels[members],
                edges_from_adjacency(adjacency),
                directed=False,
            )
            loaded_graphs.append((f"{split}-{graph_id}", graph))

    return loaded_graphs


def write_source_graph(output_folder, graph):
    """Write a normalized full graph without making any inductive split.

    The inductive builder later reads these three files and creates the compact
    training graph plus its validation/test arrays.
    """
    if graph.features is None or graph.labels is None:
        raise ValueError("a DEAL source folder needs both features and labels")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    sp.save_npz(output_folder / "A_sp.npz", graph.adjacency)
    sp.save_npz(output_folder / "X_sp.npz", graph.features)
    np.save(output_folder / "z.npy", graph.labels)


def materialize_ppi_source(source_folder, output_root):
    """Convert raw PPI files into full graph folders under ``data/PPI``.

    This is intentionally a loading operation: it only normalizes each raw
    component into Graph and writes its full A/X/z files.  It does not split
    nodes or make ``ind_train_*`` files.
    """
    for name, graph in load_ppi_graphs(source_folder):
        write_source_graph(Path(output_root) / name, graph)
        print_graph_summary(name, graph)


def print_graph_summary(name, graph):
    """Print only the details needed to confirm that a source was normalized."""
    feature_shape = None if graph.features is None else graph.features.shape
    label_shape = None if graph.labels is None else graph.labels.shape
    print(
        f"{name}: nodes={graph.num_nodes}, edges={graph.num_edges}, "
        f"directed={graph.directed}, features={feature_shape}, labels={label_shape}"
    )


if __name__ == "__main__":
    # Change this folder to inspect any saved, complete DEAL dataset.
    graph = load_deal_folder(Path(__file__).resolve().parents[2] / "data/AmazonPhoto", directed=False)
    print_graph_summary("AmazonPhoto", graph)
