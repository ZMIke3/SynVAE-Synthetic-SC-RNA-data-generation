import torch
import numbers
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler


from SynVAE.models.clvae import CLVAE
from SynVAE.losses.vaeloss import VAELoss
from SynVAE.datasets.scd import SingleCellDataset
from SynVAE.utils.utils import train_model, generate_synthetic_cells, sc_collate_fn, preprocessing
from SynVAE.visualize.visualize import visualization_preprocessing, visualize_umap, visualize_tsne, overlapping_visualization, plot_latent_space_umap


class SynVae:

    def __init__(self, adata, n_genes=None, latent_dim=30, hidden_dims=None, cell_embed_dim:int = None, trials_embed_dim: int = None, has_cell_labels:bool = False, has_trials_labels:bool = False, cells_key:str = None, trials_key:str = None, num_cell_classes=None, num_trials_labels=None):

        if adata is not None:
            self.has_cell_labels = has_cell_labels
            self.has_trials_labels = has_trials_labels
            self.dataset = SingleCellDataset(adata, has_cell_labels, has_trials_labels, cells_key, trials_key)
            self.num_trials_labels = self.dataset.len_trials_ids
            self.num_cell_classes = self.dataset.len_cell_ids
            self.cell_labels_to_id, self.trials_labels_to_id = self.dataset.get_label_to_id_dicts()
            self.cell_label_proportions, self.trials_label_proportions = self._compute_empirical_proportions()
            self.var_names = adata.var_names.copy()
            self.obs_names = adata.obs_names.copy()
            self.data = adata
        else:
            self.has_cell_labels = has_cell_labels
            self.has_trials_labels = has_trials_labels
            self.dataset = None
            self.num_trials_labels = num_trials_labels
            self.num_cell_classes = num_cell_classes
            self.cell_labels_to_id, self.trials_labels_to_id = (None, None)
            self.cell_label_proportions, self.trials_label_proportions = (None, None)
            self.var_names, self.obs_names = (None, None)
            self.dataset_len = None 
            self.data = None         

        if adata is not None:
            n_genes = adata.n_vars
        elif n_genes is None:
            raise ValueError("adata and n_genes cannot both be None")

        self.model = CLVAE(n_genes, latent_dim, hidden_dims, self.num_trials_labels, trials_embed_dim, self.num_cell_classes, cell_embed_dim)


        self.cells_key = cells_key
        self.trials_key = trials_key
        self.hidden_dims = hidden_dims

        self.cell_id_to_labels = None
        self.trials_id_to_labels = None

        if self.cell_labels_to_id is not None:
            self.cell_id_to_labels = {label_id: label for label, label_id in self.cell_labels_to_id.items()} 
            
        if self.trials_labels_to_id is not None:
            self.trials_id_to_labels = {label_id: label for label, label_id in self.trials_labels_to_id.items()}

        self.collate_fn = sc_collate_fn
        self.loss_fn = VAELoss
        self.train_fn = train_model
        self.gen_data_fn = generate_synthetic_cells
        self.data_preprocess_fn = preprocessing
        self.visualization_preprocess_fn = visualization_preprocessing
        self.visualize_umap_fn = visualize_umap
        self.visualize_tsne_fn = visualize_tsne
        self.visualize_overlap_fn = overlapping_visualization
        self.visualize_latent_space_umap_fn = plot_latent_space_umap

        self.syn_data = None
        self.sc_library_mu = None
        self.sc_library_std = None
        self.model_loss_history = None
    
    def save(self, filepath: str):
        checkpoint = {
        "config": {
            "n_genes": self.model.n_genes,
            "latent_dim": self.model.latent_dim,
            "num_trials_labels": self.num_trials_labels,
            "trials_embed_dim": self.model.trials_embed_dim,
            "num_cell_classes": self.num_cell_classes,
            "cell_embed_dim": self.model.cell_embed_dim,
            "hidden_dims": self.hidden_dims,
            "has_cell_labels": self.has_cell_labels,
            "has_trials_labels": self.has_trials_labels,
            "cells_key": self.cells_key,
            "trials_key": self.trials_key,
        },

        "state_dict": self.model.state_dict(),

        "library_mu": self.sc_library_mu,
        "library_std": self.sc_library_std,

        "var_names": None if self.var_names is None else list(self.var_names),
        "obs_names": None if self.obs_names is None else list(self.obs_names),

        "cell_labels_to_id": self.cell_labels_to_id,
        "cell_id_to_labels": self.cell_id_to_labels,

        "trials_labels_to_id": self.trials_labels_to_id,
        "trials_id_to_labels": self.trials_id_to_labels,
        "cell_label_proportions": self.cell_label_proportions,
        "trials_label_proportions": self.trials_label_proportions,
        }

        torch.save(checkpoint, filepath)

    def saveOnnx(self, input_shape, filepath: str = "SynVAE_Model.onnx"):
        self.model.eval()
        input_tensor = torch.rand(input_shape)

        torch.onnx.export(self.model,
                      input_tensor,
                      filepath,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=False,
                      input_names=['modelInput'],
                      output_names=['modelOutput'],
                      dynamic_axes={'modelInput' :{0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}})

        print(" ")
        print('Model has been converted to ONNX')

    @classmethod
    def load(cls, filepath, map_location="cpu"):

        checkpoint = torch.load(filepath, map_location=map_location)

        config = checkpoint["config"]

        model = cls(
            adata=None,
            n_genes = config["n_genes"],

            latent_dim = config["latent_dim"],
            hidden_dims = config["hidden_dims"],

            trials_embed_dim = config["trials_embed_dim"],
            cell_embed_dim = config["cell_embed_dim"],
            
            num_cell_classes=config["num_cell_classes"], 
            num_trials_labels=config["num_trials_labels"],
        )
        
        model.num_cell_classes = config["num_cell_classes"]
        model.num_trials_labels = config["num_trials_labels"]

        model.has_cell_labels = config["has_cell_labels"]
        model.has_trials_labels = config["has_trials_labels"]
        model.cells_key = config["cells_key"]
        model.trials_key = config["trials_key"]
        

        model.model.load_state_dict(checkpoint["state_dict"])

        model.sc_library_mu = checkpoint["library_mu"]
        model.sc_library_std = checkpoint["library_std"]

        model.var_names = checkpoint["var_names"]
        model.obs_names = checkpoint["obs_names"]

        model.cell_labels_to_id = checkpoint["cell_labels_to_id"]
        model.cell_id_to_labels = checkpoint["cell_id_to_labels"]

        model.trials_labels_to_id = checkpoint["trials_labels_to_id"]
        model.trials_id_to_labels = checkpoint["trials_id_to_labels"]

        model.cell_label_proportions = checkpoint["cell_label_proportions"]
        model.trials_label_proportions = checkpoint["trials_label_proportions"]
        return model

    def get_model(self):
        return self.model
    
    def get_collate_fn(self):
        return self.collate_fn
    
    def get_library_stats(self):
        return self.sc_library_mu, self.sc_library_std
    
    def id_is_defined(self, label_id, id_type):
        if id_type == "cell":
            return label_id in self.cell_id_to_labels
        elif id_type == "trials":
            return label_id in self.trials_id_to_labels
        else:
            raise ValueError("id_type must be 'cell' or 'exp'")
        
    def label_is_defined(self, label, label_type):
        if label_type == "cell":
            return label in self.cell_labels_to_id
        elif label_type == "trials":
            return label in self.trials_labels_to_id
        else:
            raise ValueError("label_type must be 'cell' or 'exp'")

    def preprocess(self, data=None, preprocess_fn=None):

        if data is None:
            data = self.data

        if not isinstance(data, sc.AnnData):
            raise ValueError("Preprocess requires Scanpy Anndata object")

        if preprocess_fn:
            data = preprocess_fn(data)
            self.dataset = SingleCellDataset(data, self.has_cell_labels, self.has_trials_labels, self.cells_key, self.trials_key)
            return self
        
        data = self.data_preprocess_fn(data)
        self.dataset = SingleCellDataset(data, self.has_cell_labels, self.has_trials_labels, self.cells_key, self.trials_key)
        return self

    def _compute_empirical_proportions(self):
        if self.has_cell_labels:
            labels, counts = np.unique(self.dataset.cell_labels, return_counts=True)
            self.cell_label_proportions = dict(zip(labels, counts / counts.sum()))
        else:
            self.cell_label_proportions = None

        if self.has_trials_labels:
            labels, counts = np.unique(self.dataset.trials_labels, return_counts=True)
            self.trials_label_proportions = dict(zip(labels, counts / counts.sum()))
        else:
            self.trials_label_proportions = None
        
        return self.cell_label_proportions, self.trials_label_proportions

    def _sample_empirical_labels(self, n_cells, label_type: str):
        if label_type == "cell":
            proportions = self.cell_label_proportions
        elif label_type == "trials":
            proportions = self.trials_label_proportions
        else:
            raise ValueError("label_type must be 'cell' or 'trials'")

        if proportions is None:
            return None

        labels = np.array(list(proportions.keys()))
        probs = np.array(list(proportions.values()))

        sampled = np.random.choice(labels, size=n_cells, p=probs)
        return sampled.tolist()

    def fit(self, batch_size=250, optimizer=None, epochs=250, lr=1e-3, vae_loss_fn=None, enc_log1p=False, use_sampler=False, num_workers=8, kl_weight=0.5, kl_annealing=False, early_stopping=True, min_delta = 1e-4, patience = 10):
        
        if self.dataset is None:
            raise RuntimeError("No dataset attached. Call preprocess(data) before fit().")

        if use_sampler:
            if not self.has_cell_labels:
                raise ValueError("use_sampler=True requires cell labels (cells_key must have been provided).")

            cell_ids_array = np.array([self.dataset.cell_labels_to_id[label] for label in self.dataset.cell_labels])
            class_counts = np.bincount(cell_ids_array)
            weights = 1.0 / class_counts[cell_ids_array]
            sampler = WeightedRandomSampler(weights, num_samples=len(self.dataset), replacement=True)

            dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler, collate_fn=self.collate_fn, num_workers=num_workers)
        else:
            dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=num_workers)


        if optimizer is None:
            optimizer = Adam(self.model.parameters(), lr=lr)

        if vae_loss_fn is None:
            vae_loss_fn = self.loss_fn

        losses, rec_losses, kl_losses, lib_mu, lib_std = self.train_fn(self.model, dataloader, optimizer, epochs, vae_loss_fn, kl_weight, enc_log1p, kl_annealing, early_stopping, min_delta, patience)

        self.sc_library_mu = lib_mu
        self.sc_library_std = lib_std
        
        self.model_loss_history = {"overall_loss": losses, "rec_loss": rec_losses, "kl_loss": kl_losses}

        return self

    def sample(self, n_cells, cell_labels=None, trials_labels=None, match_dataset_proportions=False, return_as_anndata=True):

        if self.sc_library_mu is None or self.sc_library_std is None:
                raise RuntimeError("Model must be trained before generating cells.")
        
        if n_cells <= 0:
            raise ValueError("n_cells must be positive")
        
        if cell_labels is None and match_dataset_proportions:
            cell_labels = self._sample_empirical_labels(n_cells, "cell")

        if trials_labels is None and match_dataset_proportions:
            trials_labels = self._sample_empirical_labels(n_cells, "trials")
        
        def _process_labels(labels, label_type:str):
            ids = None
            field = "cell_labels" if label_type == "cell" else "trials_labels"

            if labels is not None and isinstance(labels, np.ndarray):
                labels = labels.tolist()

            if labels is not None:
                if isinstance(labels, list) and len(labels) == 0:
                    raise ValueError(f"{field} must not be empty")
                
                if isinstance(labels, str):
                    if not self.label_is_defined(labels, label_type):
                        raise ValueError(f"{field} is not defined")
                        
                    labels = [labels] * n_cells

                    if label_type == "cell":
                        ids = [self.cell_labels_to_id[label] for label in labels]
                    elif label_type == "trials":
                        ids = [self.trials_labels_to_id[label] for label in labels]

                    ids = torch.tensor(ids, dtype=torch.int64)

                elif isinstance(labels, numbers.Integral) and not isinstance(labels, bool):
                    if not self.id_is_defined(labels, label_type):
                        raise ValueError(f"{field} is not defined")
                        
                    labels = [labels] * n_cells
                    ids = torch.tensor(labels, dtype=torch.int64)
  
                elif isinstance(labels, list):

                    if all(isinstance(x, str) for x in labels):

                        if len(labels) != n_cells:
                            raise ValueError(f"{field} must have the same length as n_cells")
                              
                        for i in range(len(labels)):
                            if not self.label_is_defined(labels[i], label_type):
                                raise ValueError(f"{field} is not defined")

                        if label_type == "cell":
                            ids = [self.cell_labels_to_id[label] for label in labels]
                        elif label_type == "trials":
                            ids = [self.trials_labels_to_id[label] for label in labels]

                        ids = torch.tensor(ids, dtype=torch.int64)

                    elif all(isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in labels):

                        if len(labels) != n_cells:
                            raise ValueError(f"{field} must have the same length as n_cells")
                            
                        for i in range(len(labels)):
                            if not self.id_is_defined(labels[i], label_type):
                                raise ValueError(f"{field} is not defined")
                                
                        ids = torch.tensor(labels, dtype=torch.int64)

                    else:
                        raise ValueError(f"{field} must be homogeneous")
                        
            return ids, labels
        
        cell_ids, cell_labels = _process_labels(cell_labels, "cell")
        exp_ids, trials_labels = _process_labels(trials_labels, "trials")

        if self.model.avg_cell_embedding is None or self.model.avg_exp_embedding is None:
            self.model.compute_avg_embedding()

        counts = self.gen_data_fn(self.model, n_cells, self.sc_library_mu, self.sc_library_std, cell_ids, exp_ids).detach().cpu().numpy()

        if return_as_anndata:
            samples = sc.AnnData(X=counts)
            samples.var_names = self.var_names.copy()
            samples.obs_names = [f"SynCell_{i}" for i in range(n_cells)]

            if trials_labels is not None:
                if all(isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in trials_labels):
                    samples.obs['experiment'] = [self.trials_id_to_labels[id_key] for id_key in exp_ids.tolist()]
                elif all(isinstance(x, str) for x in trials_labels):
                    samples.obs['experiment'] = trials_labels

            if cell_labels is not None:
                if all(isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in cell_labels):
                    samples.obs['cell_type'] = [self.cell_id_to_labels[id_key] for id_key in cell_ids.tolist()]
                elif all(isinstance(x, str) for x in cell_labels):
                    samples.obs['cell_type'] = cell_labels

            self.syn_data = samples
            return samples
        
        return counts

    def saveSynData(self, name="syndata_synvae.h5ad"):
        if self.syn_data is None:
            raise RuntimeError("Model must be sampled before saving sampled data.")
        
        self.syn_data.write_h5ad(name)

        return self

    def plotLoss(self):

        loss = self.model_loss_history['overall_loss']
        rec_loss = self.model_loss_history['rec_loss']
        kl_loss = self.model_loss_history['kl_loss']

        figs, ax = plt.subplots(1, 3, figsize=(15, 3))

        epochs = range(1, len(loss) + 1)

        ax[0].plot(epochs, loss, marker='o', linewidth=2, markersize=4, label="Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Training Loss")
        if len(loss) <= 20:
            ax[0].set_xticks(epochs)
        ax[0].grid(True, linestyle="--", alpha=0.6)
        ax[0].legend()

        ax[1].plot(epochs, rec_loss, marker='o', linewidth=2, markersize=4, label="Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].set_title("Reconstruction Loss")
        if len(rec_loss) <= 20:
            ax[1].set_xticks(epochs)
        ax[1].grid(True, linestyle="--", alpha=0.6)
        ax[1].legend()

        ax[2].plot(epochs, kl_loss, marker='o', linewidth=2, markersize=4, label="Loss")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Loss")
        ax[2].set_title("KL Loss")
        if len(kl_loss) <= 20:
            ax[2].set_xticks(epochs)
        ax[2].grid(True, linestyle="--", alpha=0.6)
        ax[2].legend()
        
        
        plt.tight_layout()
        plt.show()

    def umap(self, color='leiden', save_name=None, title='Umap Visualization', xlabel='UMAP Coordinate 1', ylabel='UMAP Coordinate 2', preprocess_fn=None):
        if self.syn_data is None:
            raise RuntimeError("Model must be sampled before visualizing data.")
            
        if preprocess_fn is None:
            data = self.visualization_preprocess_fn(self.syn_data)
            fig, ax =  self.visualize_umap_fn(data, color, title, xlabel, ylabel, save_name)
            plt.show()

        else:
            data = preprocess_fn(self.syn_data.copy())
            fig, ax =  self.visualize_umap_fn(data, color, title, xlabel, ylabel, save_name)
            plt.show()

    def tsne(self, color='leiden', save_name=None, title='Tsne Visualization', xlabel='Tsne Coordinate 1', ylabel='Tsne Coordinate 2', preprocess_fn=None):
        if self.syn_data is None:
            raise RuntimeError("Model must be sampled before visualizing data.")
        
        if preprocess_fn is None:
            data = self.visualization_preprocess_fn(self.syn_data)
            fig, ax =  self.visualize_tsne_fn(data, color, title, xlabel, ylabel, save_name)
            plt.show()

        else:
            data = preprocess_fn(self.syn_data.copy())
            fig, ax =  self.visualize_tsne_fn(data, color, title, xlabel, ylabel, save_name)
            plt.show()

    def overlapVisualization(self, adata, type="umap", color='leiden', title='Overlap Visualization', xlabel='Overlap Coordinate 1', ylabel='Overlap Coordinate 2', save_name=None, preprocess_fn=None):
        if self.syn_data is None:
            raise RuntimeError("Model must be sampled before visualizing data.")
        
        if preprocess_fn is None:
            fig, ax =  self.visualize_overlap_fn(adata, self.syn_data, type, self.visualization_preprocess_fn, self.visualize_umap_fn, self.visualize_tsne_fn, color, title, xlabel, ylabel, save_name)
            plt.show()
        else:
            # adata = preprocess_fn(adata)
            # syn_data = preprocess_fn(self.syn_data.copy())
            fig, ax =  self.visualize_overlap_fn(adata, self.syn_data, type, preprocess_fn, self.visualize_umap_fn, self.visualize_tsne_fn, color, title, xlabel, ylabel, save_name)
            plt.show()

    def plotlatent(self, dataloader, label_names=None, n_samples=50000, enc_log1p=False):
     
        if dataloader is None:
            raise ValueError("DataLoader can not be None")
        elif n_samples <= 0:
            raise ValueError("n_samples must be greater than 0")
        
        fig, ax, latent, labels = self.visualize_latent_space_umap_fn(self.model, dataloader, label_names, n_samples, enc_log1p)
        plt.show()
        
