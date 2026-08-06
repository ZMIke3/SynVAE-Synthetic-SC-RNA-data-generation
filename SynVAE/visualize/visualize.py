import scanpy as sc
import torch
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns



def plot_latent_heatmap(z_means, labels, label_names=None):

    df = pd.DataFrame(z_means)

    if label_names is not None:
        df["Cell Type"] = [label_names[label] for label in labels]

    else:
        df["Cell Type"] = labels

    mean_latent = df.groupby("Cell Type").mean()

    plt.figure(figsize=(14, 6))

    sns.heatmap(
        mean_latent,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.xlabel("Latent Dimension")
    plt.ylabel("Cell Type")
    plt.title("Average Latent Representation per Cell Type")

    plt.tight_layout()
    plt.show()

def plot_latent_space_umap(model, dataloader, label_names=None, n_samples=20000, enc_log1p=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    latent = []
    labels = []

    with torch.no_grad():
        n = 0

        for data, label, exp_id in dataloader:
            data = data.to(device)
            
            if cell_id is not None:
                cell_id = cell_id.to(device)
            if exp_id is not None:
                exp_id = exp_id.to(device)

            _, mu_z, _, _, _ = model(data, cell_ids=label, exp_ids=exp_id, enc_log1p=enc_log1p)
            
            latent.append(mu_z.cpu().numpy())
            if label is not None:
                labels.append(label.cpu().numpy())


            n += data.size(0)
            if n >= n_samples:
                break

    latent = np.concatenate(latent, axis=0)[:n_samples]
    labels = np.concatenate(labels, axis=0)[:n_samples] if labels else None

    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.3,
        metric="euclidean",
        random_state=42,
        n_components=3
    )

    embedding = reducer.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d")

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=6, alpha=0.7, color="tab:blue")
        #ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 1], s=6, alpha=0.7, color="tab:blue")
    else:
        unique_labels = np.unique(labels)

        cmap = plt.get_cmap("tab20", len(unique_labels))

        for i, lab in enumerate(unique_labels):
            mask = labels == lab

            if label_names is not None:
                ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=6,
                alpha=0.7,
                color=cmap(i),
                label=label_names[lab])
            else:
                ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=6,
                alpha=0.7,
                color=cmap(i),
                label=str(lab))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of VAE Latent Space")

    if label is not None:
        ax.legend(title="Cell Type", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()

    return fig, ax, latent, labels

def visualization_preprocessing(adata, target_sum=1e4, n_top_genes=2000, n_neighbors=30, n_pcs=15, resolution=0.5):
    adata = adata.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")

    adata.raw = adata

    adata = adata[:, adata.var.highly_variable]
    
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.umap(adata)

    return adata

def visualize_umap(adata, color, title, xlabel, ylabel, save_name=None):
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
        
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sc.pl.umap(adata, color=color, ax=ax, show=False)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Plot successfully saved to: {save_name}")

    return fig, ax

def visualize_tsne(adata, color, title, xlabel, ylabel, save_name=None):
    if 'X_tsne' not in adata.obsm:
        sc.tl.tsne(adata)
        
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sc.pl.tsne(adata, color=color, ax=ax, show=False)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Plot successfully saved to: {save_name}")

    return fig, ax

def overlapping_visualization(adata1, adata2, type, preprocess_fn, umap_fn, tsne_fn, color, title, xlabel, ylabel, save_name=None):
    adata1 = adata1.copy()
    adata2 = adata2.copy()

    adata1.obs["Plot_Label"] = 'First'
    adata2.obs["Plot_Label"] = 'Second'

    combined = sc.concat({"First":adata1, "Second":adata2}, join="outer", label='Plot_Label')

    combined = preprocess_fn(combined)

    if type == "umap":
        fig, ax  =  umap_fn(combined, color, title, xlabel, ylabel, save_name=save_name)
        plt.show()
    elif type == "tsne":
        fig, ax = tsne_fn(combined, color, title, xlabel, ylabel, save_name=save_name)
        plt.show()
    else:
        raise ValueError(f"Type parameter must be umap or tsne, got {type}")

