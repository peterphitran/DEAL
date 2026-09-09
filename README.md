# DEAL

This is the implementation of "[Inductive Link Prediction for Nodes Having Only Attribute Information
](https://www.ijcai.org/Proceedings/2020/168)" published at IJCAI 2020.

## Fork additions

This fork extends the original implementation with an inductive dataset
workflow and support for Amazon, PPI, ego-Facebook, and directed ego-Twitter.

- Sparse source and compact training artifacts for large feature matrices.
- Directed ego-Twitter splits, oriented negative examples, an asymmetric
  scorer, and a memory-mapped U8 hop-distance cache.
- PPI support as multiple independent graph components.
- Dataset loading, inductive-building, and validation tools in
  `peter_learning/` and `tools/`.

### Environment

#### Fork environment

This fork was run with a modern GPU environment:

- Python 3.10.20
- PyTorch built with CUDA 12.8
- A PyTorch Geometric version compatible with the installed PyTorch/CUDA build
- NumPy, SciPy, scikit-learn, matplotlib, and tqdm

Use a Conda environment or another isolated Python environment.  Confirm that
CUDA is visible before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

#### Original environment

The original implementation used:

- PyTorch 1.4.0
- Python 3.7.4
- PyTorch Geometric 1.4.3
- CUDA 10.1

### Train DEAL
`python train.py`

### Inductive datasets in this fork

This fork adds inductive artifact support for Amazon, PPI, ego-Facebook, and
ego-Twitter.  Source graphs and generated artifacts live under `data/` and are
not included in the repository.

#### ego-Twitter

ego-Twitter is directed, so it must use the order-sensitive `all` scorer and
the sparse feature path:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset ego-twitter --ind true --gpu --cuda 0 --mode all
```

Before training, `data/ego-twitter/` must contain the generated directed
distance cache `dists-1.u8.npy`.  The cache stores compact hop distances and
is memory-mapped so the full matrix is not moved to GPU memory.

#### Amazon Photo and Amazon Computers

The Amazon co-purchase graphs are undirected and use the standard cosine
scorer.  Replace the dataset name to choose a graph:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset AmazonPhoto --ind true --gpu --cuda 0 --mode cos

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset AmazonComputers --ind true --gpu --cuda 0 --mode cos
```

Each dataset folder needs its sparse source artifacts, inductive artifacts,
and generated `dists-1.npy` cache.

#### ego-Facebook

ego-Facebook is also an undirected graph and uses the standard cosine scorer:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset ego-facebook --ind true --gpu --cuda 0 --mode cos
```

Its dataset folder likewise needs the sparse source artifacts, inductive
artifacts, and `dists-1.npy` cache.

#### PPI

PPI consists of separate graph components.  Train one component at a time:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset PPI/train-0 --ind true --gpu --cuda 0 --mode cos
```

#### Dataset preparation tools

The `peter_learning/` directory keeps the dataset preparation workflow
separate from model training:

- `loading_datasets.py` normalizes Amazon, SNAP-style, PPI, and saved DEAL
  sources into a common `Graph` format.
- `building_inductive_dataset.py` creates node splits, edge buckets, compact
  training graphs, evaluation negatives, and DEAL artifacts.

### Datasets
More datasets can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html.

### Cite

Please cite our IJCAI 2020 paper:

```
@inproceedings{ijcai2020-168,
  title     = {Inductive Link Prediction for Nodes Having Only Attribute Information},
  author    = {Hao, Yu and Cao, Xin and Fang, Yixiang and Xie, Xike and Wang, Sibo},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {1209--1215},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/168},
  url       = {https://doi.org/10.24963/ijcai.2020/168},
}
```
