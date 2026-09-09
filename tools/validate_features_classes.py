"""Validate dataset feature dimensions and class-label structure."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from common import (AMAZON_DATASETS, PPI, csr_from_npz, download,
                    ensure_ppi_raw_dir, status)


# ---- edit these, then run: python3 contrib/tools/validate_features_classes.py ----
ROOT = "datasets/validation"   # folder where the datasets are downloaded/cached
CHECK_PUBLIC = True            # check the downloaded Amazon + PPI datasets
CHECK_DEAL = True              # check the existing DEAL-format datasets under data/
# --------------------------------------------------------------------------


AMAZON_EXPECTATIONS = {
    "Amazon Computers": {"feature_dim": 767, "classes": 10},
    "Amazon Photo": {"feature_dim": 745, "classes": 8},
}

PPI_EXPECTATIONS = {
    "feature_dim": 50,
    "classes": 121,
    "splits": {
        "train": {"nodes": 44906, "graphs": 20},
        "valid": {"nodes": 6514, "graphs": 2},
        "test": {"nodes": 5524, "graphs": 2},
    },
}

DEAL_EXPECTATIONS = {
    "CiteSeer": {
        "folder": Path("data/CiteSeer"),
        "nodes": 3327,
        "feature_dim": 3703,
        "classes": 6,
    },
}


def class_count(labels): 
    return len(np.unique(labels))


def contiguous_classes(labels, expected_classes):
    return np.array_equal(np.unique(labels), np.arange(expected_classes))


class FeatureClassValidator:
    """Checks feature dims and class labels for Amazon, PPI, and DEAL datasets."""

    def __init__(self, root):
        self.root = Path(root)

    def validate(self, include_public=True, include_deal=True):
        all_ok = True
        if include_public:
            all_ok = self._validate_amazon() and all_ok
            all_ok = self._validate_ppi() and all_ok
        if include_deal:
            all_ok = self._validate_deal() and all_ok
        return all_ok

    def _print_row(self, name, rows, expected_rows, feature_dim, expected_feature_dim, classes, expected_classes, ok):
        print(
            f"{name.ljust(18)} rows {str(rows).rjust(8)} / {str(expected_rows).ljust(8)} "
            f"features {str(feature_dim).rjust(5)} / {str(expected_feature_dim).ljust(5)} "
            f"classes {str(classes).rjust(4)} / {str(expected_classes).ljust(4)} {status(ok)}"
        )

    def _validate_amazon(self):
        all_ok = True
        for name, expected in AMAZON_EXPECTATIONS.items():
            dataset = AMAZON_DATASETS[name]
            path = self.root / dataset["file"]
            download(dataset["url"], path)

            with np.load(path) as data:
                features = csr_from_npz(data, "attr")
                labels = data["labels"]
                class_names = data["class_names"] if "class_names" in data.files else []

            rows, feature_dim = features.shape
            classes = class_count(labels)
            ok = (
                rows == dataset["nodes"]
                and feature_dim == expected["feature_dim"]
                and labels.shape == (dataset["nodes"],)
                and classes == expected["classes"]
                and contiguous_classes(labels, expected["classes"])
                and len(class_names) == expected["classes"]
            )
            self._print_row(name, rows, dataset["nodes"], feature_dim,
                            expected["feature_dim"], classes, expected["classes"], ok)
            all_ok = ok and all_ok
        return all_ok

    def _validate_ppi(self):
        raw_dir = ensure_ppi_raw_dir(self.root)
        all_ok = True
        for split, expected in PPI_EXPECTATIONS["splits"].items():
            features = np.load(raw_dir / f"{split}_feats.npy")
            labels = np.load(raw_dir / f"{split}_labels.npy")
            graph_ids = np.load(raw_dir / f"{split}_graph_id.npy")
            graph_ids = graph_ids - graph_ids.min()

            rows, feature_dim = features.shape
            label_rows, classes = labels.shape
            graphs = int(graph_ids.max()) + 1
            labels_are_binary = bool(((labels == 0) | (labels == 1)).all())
            ok = (
                rows == expected["nodes"]
                and feature_dim == PPI_EXPECTATIONS["feature_dim"]
                and label_rows == expected["nodes"]
                and classes == PPI_EXPECTATIONS["classes"]
                and graph_ids.shape == (expected["nodes"],)
                and graphs == expected["graphs"]
                and labels_are_binary
            )
            print(
                f"PPI {split.ljust(13)} rows {str(rows).rjust(8)} / {str(expected['nodes']).ljust(8)} "
                f"features {str(feature_dim).rjust(5)} / {str(PPI_EXPECTATIONS['feature_dim']).ljust(5)} "
                f"classes {str(classes).rjust(4)} / {str(PPI_EXPECTATIONS['classes']).ljust(4)} "
                f"graphs {str(graphs).rjust(3)} / {str(expected['graphs']).ljust(3)} {status(ok)}"
            )
            all_ok = ok and all_ok
        return all_ok

    def _validate_deal(self):
        all_ok = True
        for name, expected in DEAL_EXPECTATIONS.items():
            folder = expected["folder"]
            features_path = folder / "X_sp.npz"
            labels_path = folder / "z.npy"
            if not features_path.exists() or not labels_path.exists():
                print(f"{name.ljust(18)} missing {features_path} or {labels_path} FAIL")
                all_ok = False
                continue

            features = sp.load_npz(features_path)
            labels = np.load(labels_path)
            rows, feature_dim = features.shape
            classes = class_count(labels)
            ok = (
                rows == expected["nodes"]
                and feature_dim == expected["feature_dim"]
                and labels.shape == (expected["nodes"],)
                and classes == expected["classes"]
                and contiguous_classes(labels, expected["classes"])
            )
            self._print_row(name, rows, expected["nodes"], feature_dim,
                            expected["feature_dim"], classes, expected["classes"], ok)
            all_ok = ok and all_ok
        return all_ok


def main():
    validator = FeatureClassValidator(ROOT)
    ok = validator.validate(include_public=CHECK_PUBLIC, include_deal=CHECK_DEAL)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())