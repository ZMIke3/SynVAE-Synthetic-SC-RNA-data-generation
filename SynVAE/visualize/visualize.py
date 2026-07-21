import scanpy as sc
import matplotlib.pyplot as plt

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

def visualize_heatmap(adata, color, title, xlabel, ylabel, save_name=None):
    adata = adata.copy()

    fig, ax = plt.subplots(figsize=(6, 5))

    sc.pl.heatmap(adata, var_names=adata.var_names, groupby=color, ax=ax, show=False)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Plot successfully saved to: {save_name}")

    return fig, ax

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
    print("here4")

    if type == "umap":
        fig, ax  =  umap_fn(combined, color, title, xlabel, ylabel, save_name=None)
        plt.show()
    elif type == "tsne":
        fig, ax = tsne_fn(combined, color, title, xlabel, ylabel, save_name=None)
        plt.show()
    else:
        raise ValueError(f"Type parameter must be umap or tsne, got {type}")

