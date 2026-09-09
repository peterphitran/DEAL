from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


DATASET = "ego-facebook"  # AmazonPhoto, AmazonComputers, ego-facebook, ego-twitter, or PPI
BATCH_SIZE = 128


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TWITTER_DATASET = "ego-twitter"
PPI_DATASET = "PPI"

FLOAT_CACHE_NAME = "dists-1.npy"
UINT8_CACHE_NAME = "dists-1.u8.npy"

SUPPORTED_DATASETS = {
    "AmazonPhoto",
    "AmazonComputers",
    "ego-facebook",
    TWITTER_DATASET,
    PPI_DATASET,
}


def load_training_adjacency(dataset_dir):
    """Load and validate one compact training adjacency matrix."""
    adjacency_path = Path(dataset_dir) / "ind_train_A.npz"
    if not adjacency_path.is_file():
        raise FileNotFoundError(f"missing {adjacency_path}")

    adjacency = sp.load_npz(adjacency_path).tocsr()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"{adjacency_path} must be square")
    return adjacency


def encode_distances(path_distances, uint8_hops):
    """Encode one batch of shortest-path distances for its cache format."""
    reachable = np.isfinite(path_distances)

    if uint8_hops:
        if reachable.any() and path_distances[reachable].max() >= 255:
            raise ValueError("a hop count cannot be stored in uint8")

        encoded = np.zeros(path_distances.shape, dtype=np.uint8)
        encoded[reachable] = path_distances[reachable].astype(np.uint8) + 1
        return encoded

    encoded = np.zeros(path_distances.shape, dtype=np.float32)
    encoded[reachable] = 1.0 / (path_distances[reachable] + 1.0)
    return encoded


def build_distance_cache_for_folder(dataset_dir, directed=False, uint8_hops=False):
    """Write one compact training graph's distance cache.

    Standard datasets store float32 transformed distances.  Directed Twitter
    stores exact hop counts as uint8 so training can decode only its current
    batch without moving an all-pairs matrix to the GPU.
    """
    dataset_dir = Path(dataset_dir)
    adjacency = load_training_adjacency(dataset_dir)
    node_count = adjacency.shape[0]

    cache_name = UINT8_CACHE_NAME if uint8_hops else FLOAT_CACHE_NAME
    cache_dtype = np.uint8 if uint8_hops else np.float32
    output_path = dataset_dir / cache_name
    temporary_path = dataset_dir / cache_name.replace(".npy", ".tmp.npy")
    # temporary_path prevents an incomplete cache 
    # from looking like a finished cache

    cache = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=cache_dtype,
        shape=(node_count, node_count),
    )

    for start in range(0, node_count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, node_count)
        source_nodes = np.arange(start, stop)
        path_distances = dijkstra(
            adjacency,
            directed=directed,
            indices=source_nodes,
        )
        path_distances = np.atleast_2d(path_distances)
        cache[start:stop] = encode_distances(path_distances, uint8_hops)
        print(f"sources {start} to {stop - 1}")

    cache.flush()
    del cache # Release the memory-mapped file before renaming it.
    temporary_path.replace(output_path)
    print(
        f"cache {output_path} shape {(node_count, node_count)} "
        f"dtype {cache_dtype}"
    )


def find_ppi_graph_folders():
    """Return every PPI folder containing a compact training graph."""
    ppi_dir = DATA_ROOT / PPI_DATASET
    if not ppi_dir.is_dir():
        raise FileNotFoundError(f"missing PPI data folder {ppi_dir}")

    graph_folders = sorted(
        folder
        for folder in ppi_dir.iterdir()
        if folder.is_dir() and (folder / "ind_train_A.npz").is_file()
    )
    if not graph_folders:
        raise FileNotFoundError(f"no PPI inductive graphs found in {ppi_dir}")
    return graph_folders


def build_distance_cache(dataset):
    """Write compact distance caches for one dataset or every PPI component."""
    if dataset not in SUPPORTED_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"DATASET must be one of {choices}")

    if dataset == PPI_DATASET:
        for folder in find_ppi_graph_folders():
            print(f"PPI graph {folder.name}")
            build_distance_cache_for_folder(folder)
        return

    is_twitter = dataset == TWITTER_DATASET
    build_distance_cache_for_folder(
        DATA_ROOT / dataset,
        directed=is_twitter,
        uint8_hops=is_twitter,
    )


def main():
    build_distance_cache(DATASET)


if __name__ == "__main__":
    main()
