# Repository for "Implicit degree bias in the link prediction task"

## Citation
```
@article{,
  title={},
  author={},
  journal={arxiv: xxxx}
  year={}
}
```

## Installing the packages

We recommend using Miniforge [mamba](https://github.com/conda-forge/miniforge) to manage the packages.

Specifically, we build the conda environment with the following command.
```bash
mamba create -n det-lim-weighted -c bioconda -c nvidia -c pytorch -c pyg python=3.11 cuda-version=12.1 pytorch torchvision torchaudio pytorch-cuda=12.1 snakemake graph-tool scikit-learn numpy==1.23.5 numba scipy==1.10.1 pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu pyg pytorch-sparse python-igraph -y
pip install adabelief-pytorch==0.2.0
pip install GPUtil powerlaw leidenalg
```

Additionally, we need the following custom packages to run the experiments.
- [gnn_tools](https://github.com/skojaku/gnn-tools) provides the code for generating graph embeddings using the GNNs. We used [the version 1.0](https://github.com/skojaku/gnn-tools/releases/tag/v1.0)
- [embcom](https://github.com/skojaku/embcom) provides supplementary graph embedding methods. We used [the version 1.01](https://github.com/skojaku/embcom/releases/tag/v1.01)
- [LFR-benchmark](https://github.com/skojaku/LFR-benchmark) provides the code for the LFR benchmark. We used [version 1.01](https://github.com/skojaku/LFR-benchmark/releases/tag/v1.01).

These packages can be installed via pip as follows:
```bash
pip install git+https://github.com/skojaku/gnn-tools.git@v1.0
pip install git+https://github.com/skojaku/embcom.git@v1.01
```
And to install the LFR benchmark package:
```bash
git clone https://github.com/skojaku/LFR-benchmark
cd LFR-benchmark
python setup.py build
pip install -e .
```

## Running the experiments

We provide the snakemake file to run the experiments. Before running the snakemake, you must create a `config.yaml` file under the `workflow/` directory.
```yaml
data_dir: "data/"
```
where `data_dir` is the directory where all data will is located


Once you have created the `config.yaml` file, run the snakemake as follows:
```bash
snakemake --cores <number of cores> all
```
or conveniently,
```bash
nohup snakemake --cores <number of cores> all >log &
```
The Snakemake will preprocess the data, run the experiments, and generate the figures in `reproduction/figs/` directory.
