"""Shared registries and helpers for the contrib/tools/ dataset scripts.

Everything here is NumPy + SciPy only, so it runs on base WSL python3 without
the legacy Torch/PyG stack. The other tools import from this module instead of
duplicating the dataset URLs, the downloader, and the sparse-matrix loader.
"""

import urllib.request
import zipfile

import scipy.sparse as sp


# --- dataset registries --------------------------------------------------

SNAP_DATASETS = {
    "ego-facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "file": "facebook_combined.txt.gz",
        "nodes": 4039,
        "edges": 88234,
        "unique_undirected": True,
    },
    "ego-twitter": {
        "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
        "file": "twitter_combined.txt.gz",
        "nodes": 81306,
        "edges": 1768149,
        "unique_undirected": False,
        "unique_directed": True,
    },
}

AMAZON_DATASETS = {
    "Amazon Computers": {
        "url": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz",
        "file": "amazon_electronics_computers.npz",
        "nodes": 13752,
        "edges": 491722,
    },
    "Amazon Photo": {
        "url": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_photo.npz",
        "file": "amazon_electronics_photo.npz",
        "nodes": 7650,
        "edges": 238162,
    },
}

PPI = {
    "url": "https://data.dgl.ai/dataset/ppi.zip",
    "file": "ppi.zip",
    "graphs": 20,
    "avg_nodes": 2245.3,
    "avg_edges": 61318.4,
}


# --- shared helpers ------------------------------------------------------

def download(url, path):
    """Download `url` to `path` once; skip if the file already exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {path.name}")
    urllib.request.urlretrieve(url, path)


def status(ok):
    """Render a boolean check result as a fixed OK / FAIL token."""
    return "OK" if ok else "FAIL"


def csr_from_npz(data, prefix):
    """Rebuild a CSR matrix saved under `<prefix>_data/indices/indptr/shape`."""
    return sp.csr_matrix(
        (data[f"{prefix}_data"], data[f"{prefix}_indices"], data[f"{prefix}_indptr"]),
        shape=data[f"{prefix}_shape"],
    )


def ensure_ppi_raw_dir(root):
    """Download and extract ppi.zip, returning the folder holding the .npy files."""
    archive_path = root / PPI["file"]
    download(PPI["url"], archive_path)

    extract_dir = root / "ppi"
    nested_dir = extract_dir / "ppi"
    if not (extract_dir / "train_graph.json").exists() and not (nested_dir / "train_graph.json").exists():
        print("extract ppi.zip")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

    return nested_dir if nested_dir.exists() else extract_dir
