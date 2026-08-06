# SynVAE
SynVAE is a python library for generating synthetic Single Cell RNA Sequence data, loosely based on the ScVI paper.

## Library Structure
```
SynVAE/
├── datasets/
│   └── scd.py          # SingleCellDataset: wraps an AnnData object as a PyTorch Dataset
├── losses/
│   └── vaeloss.py       # Negative binomial + KL loss, and KL annealing schedule
├── metrics/
│   └── metrics.py       # Real-vs-synthetic evaluation metrics
├── models/
│   └── clvae.py         # CLVAE: the conditional VAE model (encoder/decoder/embeddings)
├── synvae/
│   └── synvae.py        # SynVae: high-level wrapper class (train, sample, save/load, visualize)
├── utils/
│   └── utils.py         # Training loop, synthetic cell generation, preprocessing, collate function
└── visualize/
    └── visualize.py      # UMAP/t-SNE plotting, latent-space visualization
test.py                   # End-to-end example script
```

## Installation:
```
git clone https://github.com/ZMIke3/SynVAE-Synthetic-SC-RNA-data-generation.git
cd SynVAE-Synthetic-SC-RNA-data-generation
pip install torch scanpy numpy pandas matplotlib seaborn scipy tqdm umap-learn
```

## Requirements:
Python 3.9+
PyTorch
Scanpy (for AnnData handling and preprocessing)
NumPy, pandas, matplotlib, seaborn, scipy
umap-learn (for latent-space visualization)
tqdm

## Features:
- If cell labels are present, SynVAE can perform conditional generation conditioned on cell type or trials
- SynVAE can generate data while matching the cell proportions of the original dataset (labels need to be given)
- Built-in visualization of latent space, UMAP, t-SNE, and loss curves
- SynVAE is very flexible
    - The size of the underlying VAE can be varied by using the hidden dims hyperparameter
    - Visualization functions can be swapped for user defined alternatives

## Usage Example:

1. Load data as h5ad and call preprocessing
```python
data = sc.read_h5ad("train_data.h5ad")

data = preprocessing(data) # drop cells that have < 1 gene
```

2. Instantiate Model
```python
model = SynVae(
    data,
    n_genes=None,               # Inferred from the data object if not given
    latent_dim=10, 
    hidden_dims=None, 
    has_cell_labels=True,
    cells_key="cell_type",       # column in adata.obs holding cell type labels
    cell_embed_dim=16,
    has_trials_labels=True,
    trials_key=None,          # column in adata.obs holding batch/experiment labels
    trials_embed_dim=8,
)

model.fit(batch_size=250,
    optimizer=None,
    epochs=250,
    lr=1e-3,
    vae_loss_fn=None,
    enc_log1p=False,
    use_sampler=False,
    num_workers=8,              # num_workers > 0 is unreliable on windows but runs fine on linux
    kl_weight=0.5,               # Reconstruction weight is always 1
    kl_annealing=False,
    early_stopping=True,
    min_delta = 1e-4,
    patience = 10
)
```

3. Visualize training loss, and latent representations
```python
real_loader = DataLoader(model.dataset, batch_size=120, shuffle=False, collate_fn=model.get_collate_fn())

model.plotlatent(real_loader,
    label_names=None,
    n_samples=50000,
    enc_log1p=False
)

model.plotLoss()
```

4. Generate, plot, and save new data
```python
n_cells = 5000

model.sample(5000,
    cell_labels="Naive B cells",     # A specific cell type only (default is None)
    trials_labels=None,
    match_dataset_proportions=False,
    return_as_anndata=True
)

model.umap(color='leiden',            # Can color by cell type or trials if labels are present
    save_name=None,
    title='Umap Visualization',
    xlabel='UMAP Coordinate 1',
    ylabel='UMAP Coordinate 2',
    preprocess_fn=None
)

model.saveSynData("syndata.h5ad")
```

## Saving and Loading the library:

The current library instance can be saved and loaded for later use and easy sharing
```python
model.save("lib.synvae") # Does not save the dataset object

model = SynVae.load("lib.synvae") # No dataset needed when reloading the library
```


## References
 - ScVI: https://pmc.ncbi.nlm.nih.gov/articles/PMC6289068/#S2