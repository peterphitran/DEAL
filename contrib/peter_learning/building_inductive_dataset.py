# AI dont touch this file I am using this to learn

"""
should take a full graph + features
    - split nodes
    - group edges by split creating the empty node lists
    - remap kept training nodes
        (compact needs normal contiguous matrix ID's that gives us a smaller training
        adjacency and feature matrix containing only visible training nodes.)
    - create compact training graph 
        (Mostly it is for hiding nodes and all their incident evaluation edges, 
        not merely hiding individual edges.)
    - create evaluation positives/negatives
        (sample positives and negatives to get accuracy of node predicitions)
    - validate
    - write the DEAL files 

Without bucketing, hidden-node edges could accidentally leak into training.

graph object only needs 
    - features (matrix mapping node ID to features)
    - adjacency (matrix mapping node IDs that are connected with an edge)
    - edges (source and target pairs)
    - directed (boolean for handing undirected and directed graphs)

full graph
- node split
- edge buckets
- compact node mapping
- compact training adjacency
- compact training features
- validation/test positives
- sampled negatives
- validation

sparse only stores non zero, dense stores every value including zeroes
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from pathlib import Path

p_g = .5 # nodes in graph
p_tr = .3 # nodes in training
p_va = .1 # nodes in validation
p_te = .1 # nodes in testing

ARRAYS_FILENAME = "pv0.10_pt0.00_pn0.10_arrays.npz"
ROOT = Path(__file__).resolve().parents[2]
DATASET = "toy"  # Use "toy", "PPI", or a dataset name from DATASETS below.
PPI_DATA_FOLDER = ROOT / "data/PPI"

DATASETS = {
    "CiteSeer": {
        "folder": ROOT / "data/CiteSeer",
        "directed": True,
    },
    "AmazonPhoto": {
        "folder": ROOT / "data/AmazonPhoto",
        "directed": False,
    },
    "AmazonComputers": {
        "folder": ROOT / "data/AmazonComputers",
        "directed": False,
    },
    "ego-facebook": {
        "folder": ROOT / "data/ego-facebook",
        "directed": False,
    },
    "ego-twitter": {
        "folder": ROOT / "data/ego-twitter",
        "directed": True,
    },
}

@dataclass
class Graph:
    """One full graph in the common format used by the inductive pipeline."""
    adjacency: sp.csr_matrix
    features: sp.csr_matrix
    labels: np.ndarray
    edges: np.ndarray
    directed: bool

    @property
    def num_nodes(self):
        return self.adjacency.shape[0]

def load_deal_graph(data_dir, directed):
    """Load one full DEAL graph from its saved adjacency/features/labels."""
    data_dir = Path(data_dir)

    adjacency = sp.load_npz(data_dir / "A_sp.npz").tocsr()
    features = sp.load_npz(data_dir / "X_sp.npz").tocsr()
    labels = np.load(data_dir / "z.npy")

    num_nodes = adjacency.shape[0]
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("full adjacency must be square")
    if features.shape[0] != num_nodes:
        raise ValueError("feature rows must match adjacency rows")
    if labels.ndim < 1 or labels.shape[0] != num_nodes:
        raise ValueError("labels must contain one row per node")
    if not directed and (adjacency != adjacency.T).nnz:
        raise ValueError("undirected graph must have symmetric adjacency")

    source_nodes, target_nodes = adjacency.nonzero()
    edges = np.vstack((source_nodes, target_nodes)).astype(np.int64)
    return Graph(adjacency, features, labels, edges, directed)

def load_dataset(name):
    """Load a single-graph dataset selected from DATASETS."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    dataset = DATASETS[name]
    return load_deal_graph(dataset["folder"], dataset["directed"])

def load_ppi_dataset():
    """Load each already-built PPI graph from its own data/PPI subfolder."""
    graph_folders = sorted(
        folder for folder in PPI_DATA_FOLDER.iterdir()
        if folder.is_dir() and (folder / "A_sp.npz").is_file()
    ) if PPI_DATA_FOLDER.is_dir() else []

    if not graph_folders:
        raise FileNotFoundError(
            f"no PPI artifacts in {PPI_DATA_FOLDER}; run build_ppi_artifacts() first"
        )

    return [
        (folder.name, load_deal_graph(folder, directed=False))
        for folder in graph_folders
    ]

def make_toy_graph():
    """Create the small graph used to learn and check each pipeline step."""
    num_nodes = 10
    edges = np.array([[5, 0, 6, 7, 2],
                      [6, 2, 1, 8, 1]])
    features = np.arange(num_nodes * 3).reshape(num_nodes, 3)
    adjacency = sp.csr_matrix(
        (np.ones(edges.shape[1]), (edges[0], edges[1])),
        shape=(num_nodes, num_nodes),
    )
    labels = np.zeros(num_nodes, dtype=np.int64)
    return Graph(adjacency, sp.csr_matrix(features), labels, edges, directed=True)


def split_nodes(graph, p_g, p_tr, p_va, seed):
    NG = int(graph.num_nodes * p_g)
    Ntr = int(graph.num_nodes * p_tr)
    Nv = int(graph.num_nodes * p_va)
    Nte = graph.num_nodes - (NG + Ntr + Nv)
    node_ids = np.arange(graph.num_nodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(node_ids)

    # observed node group
    VG = node_ids[:NG] 
    Vtr = node_ids[NG:NG + Ntr]
    Vv = node_ids[NG + Ntr: NG + Ntr + Nv]
    Vte = node_ids[NG + Ntr + Nv:]
    set_id = np.empty(graph.num_nodes, dtype=np.int8)

    # store code for group 
    set_id[VG] = 0
    set_id[Vtr] = 1
    set_id[Vv] = 2
    set_id[Vte] = 3

    return VG, Vtr, Vv, Vte, set_id

def bucket_edges(graph, set_id):
    buckets = {
        "EG": [],
        "Etr": [],
        "Ev": [],
        "Ete": [],
        "ignored": [],
    }

    for source, target in zip(graph.edges[0], graph.edges[1]):
        source_group = set_id[source]
        target_group = set_id[target]

        if source_group == 0 and target_group == 0:
            buckets["EG"].append((source, target))
        elif (source_group == 0 and target_group == 1) or (source_group == 1 and target_group == 0):
            buckets["Etr"].append((source, target))
        elif (source_group == 0 and target_group == 2) or (source_group == 2 and target_group == 0):
            buckets["Ev"].append((source, target))
        elif (source_group == 0 and target_group == 3) or (source_group == 3 and target_group == 0):
            buckets["Ete"].append((source, target))
        else:
            buckets["ignored"].append((source, target))
    
    return buckets

def training_graph(graph, VG, Vtr, EG, Etr):
    nodes_keep = np.union1d(VG, Vtr)
    full_to_compact = np.full(graph.num_nodes, -1, dtype=np.int64)

    for compact_id, full_node_id in enumerate(nodes_keep):
        full_to_compact[full_node_id] = compact_id

    training_edges = EG + Etr
    compact_training_edges = []

    for source, target in training_edges:
        compact_source = full_to_compact[source]
        compact_target = full_to_compact[target]

        if compact_source == -1 or compact_target == -1:
            raise ValueError("hidden node leaked into training edge")

        compact_training_edges.append((compact_source, compact_target))

    rows = []
    cols = []

    for source, target in compact_training_edges:
        rows.append(source)
        cols.append(target)

    data = np.ones(len(compact_training_edges), dtype=np.float32)

    training_adjacency = sp.csr_matrix((
        data, (rows, cols)), shape=(len(nodes_keep), len(nodes_keep))
    )

    return full_to_compact, nodes_keep, training_edges, compact_training_edges, training_adjacency

def compact_feat_matrix(graph, nodes_keep):
    X_train = graph.features[nodes_keep]
    return X_train

def eval_negative(positives, source_nodes, target_nodes, edge_set, seed):
    rng = np.random.default_rng(seed)
    negatives = []
    seen = set()

    for _ in positives:
        while True:
            source = int(rng.choice(source_nodes))
            target = int(rng.choice(target_nodes))
            candidate = (source, target)

            if source == target:
                continue

            if candidate in edge_set:
                continue
            if candidate in seen:
                continue

            negatives.append(candidate)
            seen.add(candidate)
            break

    return negatives

def eval_directed_negatives(positives, observed_nodes, hidden_nodes, edge_set, seed):
    """Match the direction of each directed evaluation positive.

    A positive in an evaluation bucket is either observed -> hidden or hidden
    -> observed.  The negative must come from the same orientation.
    """
    observed_set = set(observed_nodes.tolist())
    forward_positives = []
    reverse_positives = []

    for source, target in positives:
        if source in observed_set:
            forward_positives.append((source, target))
        else:
            reverse_positives.append((source, target))

    forward_negatives = eval_negative(
        forward_positives, observed_nodes, hidden_nodes, edge_set, seed
    )
    reverse_negatives = eval_negative(
        reverse_positives, hidden_nodes, observed_nodes, edge_set, seed + 1
    )
    return forward_negatives + reverse_negatives

def validate_dataset(
    graph,
    VG,
    Vtr,
    Vv,
    Vte,
    nodes_keep,
    full_to_compact,
    compact_training_edges,
    training_adjacency,
    val_positives,
    val_negatives,
    test_positives,
    test_negatives,
    edge_set,
    verbose=True,
):
    all_split_nodes = np.concatenate((VG, Vtr, Vv, Vte))

    if len(all_split_nodes) != graph.num_nodes:
        raise ValueError("split does not contain every node")

    if len(np.unique(all_split_nodes)) != graph.num_nodes:
        raise ValueError("a node appears in more than one split group")

    expected_keep = np.union1d(VG, Vtr)
    if not np.array_equal(nodes_keep, expected_keep):
        raise ValueError("nodes_keep must contain exactly VG and Vtr")

    if np.any(full_to_compact[Vv] != -1):
        raise ValueError("validation node leaked into compact graph")

    if np.any(full_to_compact[Vte] != -1):
        raise ValueError("test node leaked into compact graph")

    expected_shape = (len(nodes_keep), len(nodes_keep))
    if training_adjacency.shape != expected_shape:
        raise ValueError("compact adjacency has the wrong shape")

    for source, target in compact_training_edges:
        if source < 0 or target < 0:
            raise ValueError("compact training edge contains hidden node")

        if source >= len(nodes_keep) or target >= len(nodes_keep):
            raise ValueError("compact training edge is out of bounds")

    for source, target in val_positives + test_positives:
        if (source, target) not in edge_set:
            raise ValueError("evaluation positive is not a real full-graph edge")

    for source, target in val_negatives + test_negatives:
        if (source, target) in edge_set:
            raise ValueError("evaluation negative is a real full-graph edge")

    if graph.directed:
        observed_set = set(VG.tolist())
        for positives, negatives in ((val_positives, val_negatives),
                                     (test_positives, test_negatives)):
            positive_forward = sum(source in observed_set for source, _ in positives)
            negative_forward = sum(source in observed_set for source, _ in negatives)
            if positive_forward != negative_forward:
                raise ValueError("directed negative orientations do not match positives")

    if verbose:
        print("validation passed")

def as_edge_array(edges):
    """Return edge pairs in DEAL's [number of edges, 2] array shape."""
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)

def as_sparse_matrix(matrix):
    """Keep sparse inputs sparse; convert dense learning examples when needed."""
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)

def write_deal_artifacts(
    output_dir,
    graph,
    training_adjacency,
    training_features,
    nodes_keep,
    training_positives,
    val_positives,
    val_negatives,
    test_positives,
    test_negatives,
):
    """Write one inductive dataset in the file layout consumed by DEAL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sp.save_npz(output_dir / "A_sp.npz", as_sparse_matrix(graph.adjacency))
    sp.save_npz(output_dir / "X_sp.npz", as_sparse_matrix(graph.features))
    np.save(output_dir / "z.npy", np.asarray(graph.labels))

    sp.save_npz(output_dir / "ind_train_A.npz", as_sparse_matrix(training_adjacency))
    sp.save_npz(output_dir / "ind_train_X.npz", as_sparse_matrix(training_features))
    np.save(output_dir / "nodes_keep.npy", np.asarray(nodes_keep, dtype=np.int64))

    # load_datafile unpacks these arrays by their saved order.
    np.savez(
        output_dir / ARRAYS_FILENAME,
        as_edge_array(training_positives),
        as_edge_array(val_positives),
        as_edge_array(val_negatives),
        as_edge_array(test_positives),
        as_edge_array(test_negatives),
    )

def run_inductive_pipeline(graph, seed, verbose=True):
    """Apply the learning pipeline to one feature-bearing Graph."""
    if graph.features is None:
        raise ValueError("the inductive pipeline needs node features")

    edge_set = set()
    for source, target in zip(graph.edges[0], graph.edges[1]):
        edge_set.add((int(source), int(target)))

    VG, Vtr, Vv, Vte, set_id = split_nodes(graph, p_g, p_tr, p_va, seed)
    buckets = bucket_edges(graph, set_id)
    val_positives = buckets["Ev"]
    test_positives = buckets["Ete"]

    full_to_compact, nodes_keep, training_edges, compact_training_edges, training_adjacency = training_graph(graph, VG, Vtr, buckets["EG"], buckets["Etr"])
    X_train = compact_feat_matrix(graph, nodes_keep)
    if graph.directed:
        val_negatives = eval_directed_negatives(val_positives, VG, Vv, edge_set, seed)
        test_negatives = eval_directed_negatives(test_positives, VG, Vte, edge_set, seed)
    else:
        val_negatives = eval_negative(val_positives, VG, Vv, edge_set, seed)
        test_negatives = eval_negative(test_positives, VG, Vte, edge_set, seed)

    validate_dataset(
        graph,
        VG, 
        Vtr,
        Vv,
        Vte,
        nodes_keep,
        full_to_compact,
        compact_training_edges,
        training_adjacency,
        val_positives,
        val_negatives,
        test_positives,
        test_negatives,
        edge_set,
        verbose=verbose,
    )

    return {
        "buckets": buckets,
        "nodes_keep": nodes_keep,
        "compact_training_edges": compact_training_edges,
        "training_adjacency": training_adjacency,
        "training_features": X_train,
        "val_positives": val_positives,
        "val_negatives": val_negatives,
        "test_positives": test_positives,
        "test_negatives": test_negatives,
    }


def print_pipeline_summary(name, graph, result):
    """Print the key results without dumping each graph's large matrices."""
    buckets = result["buckets"]
    print(
        f"{name}: nodes={graph.num_nodes}, kept={len(result['nodes_keep'])}, "
        f"EG={len(buckets['EG'])}, Etr={len(buckets['Etr'])}, "
        f"Ev={len(buckets['Ev'])}, Ete={len(buckets['Ete'])}"
    )


def build_ppi_splits(seed=42):
    """Read full data/PPI graphs and write their complete DEAL artifacts."""
    for name, graph in load_ppi_dataset():
        result = run_inductive_pipeline(graph, seed, verbose=False)
        write_deal_artifacts(
            PPI_DATA_FOLDER / name,
            graph,
            result["training_adjacency"],
            result["training_features"],
            result["nodes_keep"],
            result["compact_training_edges"],
            result["val_positives"],
            result["val_negatives"],
            result["test_positives"],
            result["test_negatives"],
        )
        print_pipeline_summary(name, graph, result)


def main():
    if DATASET == "toy":
        graphs = [("toy", make_toy_graph())]
    elif DATASET == "PPI":
        graphs = load_ppi_dataset()
    else:
        graphs = [(DATASET, load_dataset(DATASET))]

    for name, graph in graphs:
        result = run_inductive_pipeline(graph, seed=42, verbose=DATASET == "toy")
        print_pipeline_summary(name, graph, result)


if __name__ == "__main__":
    main()
