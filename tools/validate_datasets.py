"""Validate dataset node/edge counts against their expected public sizes."""

import gzip
import json
import math
from pathlib import Path

import numpy as np

from common import (AMAZON_DATASETS, PPI, SNAP_DATASETS, csr_from_npz,
                    download, ensure_ppi_raw_dir, status)


# ---- edit this, then run: python3 contrib/tools/validate_datasets.py ----
ROOT = "datasets/validation"   # folder where the datasets are downloaded/cached
# -----------------------------------------------------------------


class DatasetSizeValidator:
    """Checks node/edge counts for the SNAP, Amazon, and PPI datasets."""

    def __init__(self, root):
        self.root = Path(root)

    def validate(self):
        all_ok = True
        all_ok = self._validate_snap() and all_ok
        all_ok = self._validate_amazon() and all_ok
        all_ok = self._validate_ppi() and all_ok
        return all_ok

    def _print_row(self, name, nodes, expected_nodes, edges, expected_edges, ok):
        print(
            f"{name.ljust(18)} nodes {str(nodes).rjust(10)} / {str(expected_nodes).ljust(10)} "
            f"edges {str(edges).rjust(10)} / {str(expected_edges).ljust(10)} {status(ok)}"
        )

    def _validate_snap(self):
        all_ok = True
        for name, info in SNAP_DATASETS.items():
            path = self.root / info["file"]
            download(info["url"], path)

            nodes = set()
            edge_count = 0
            directed_edges = set()
            undirected_edges = set()

            with gzip.open(path, "rt") as handle:
                for line in handle:
                    source_text, target_text = line.split()
                    source = int(source_text)
                    target = int(target_text)
                    nodes.add(source)
                    nodes.add(target)
                    edge_count += 1
                    if info.get("unique_directed"):
                        directed_edges.add((source, target))
                    if info["unique_undirected"]:
                        undirected_edges.add((min(source, target), max(source, target)))

            if info.get("unique_directed"):
                edges = len(directed_edges)
            elif info["unique_undirected"]:
                edges = len(undirected_edges)
            else:
                edges = edge_count
            ok = len(nodes) == info["nodes"] and edges == info["edges"]
            self._print_row(name, len(nodes), info["nodes"], edges, info["edges"], ok)
            all_ok = ok and all_ok
        return all_ok

    def _validate_amazon(self):
        all_ok = True
        for name, info in AMAZON_DATASETS.items():
            path = self.root / info["file"]
            download(info["url"], path)

            with np.load(path) as data:
                adj = csr_from_npz(data, "adj")

            adj.setdiag(0)
            adj.eliminate_zeros()
            adj = (adj + adj.T).astype(bool)
            nodes = adj.shape[0]
            edges = int(adj.nnz)
            ok = nodes == info["nodes"] and edges == info["edges"]
            self._print_row(name, nodes, info["nodes"], edges, info["edges"], ok)
            all_ok = ok and all_ok
        return all_ok

    def _validate_ppi(self):
        raw_dir = ensure_ppi_raw_dir(self.root)

        graph_ids = np.load(raw_dir / "train_graph_id.npy")
        graph_ids = graph_ids - graph_ids.min()
        graph_count = int(graph_ids.max()) + 1
        members = [set(np.flatnonzero(graph_ids == graph_number).astype(int)) for graph_number in range(graph_count)]

        node_to_graph = {}
        for graph_number, node_ids in enumerate(members):
            for node_id in node_ids:
                node_to_graph[node_id] = graph_number

        with open(raw_dir / "train_graph.json", encoding="utf-8") as handle:
            graph = json.load(handle)

        edge_counts = [0 for _ in range(graph_count)]
        for link in graph["links"]:
            source = int(link["source"])
            target = int(link["target"])
            if source == target:
                continue
            graph_number = node_to_graph.get(source)
            if graph_number is not None and graph_number == node_to_graph.get(target):
                edge_counts[graph_number] += 1

        avg_nodes = sum(len(node_ids) for node_ids in members) / graph_count
        avg_edges = sum(edge_counts) / graph_count
        ok = (
            graph_count == PPI["graphs"]
            and math.isclose(avg_nodes, PPI["avg_nodes"], abs_tol=0.05)
            and math.isclose(avg_edges, PPI["avg_edges"], abs_tol=0.05)
        )
        print(
            f"{'PPI train'.ljust(18)} graphs {str(graph_count).rjust(9)} / {str(PPI['graphs']).ljust(10)} "
            f"avg nodes {str(round(avg_nodes, 1)).rjust(8)} / {str(PPI['avg_nodes']).ljust(8)} "
            f"avg edges {str(round(avg_edges, 1)).rjust(9)} / {str(PPI['avg_edges']).ljust(9)} {status(ok)}"
        )
        return ok


def main():
    ok = DatasetSizeValidator(ROOT).validate()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
