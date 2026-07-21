import torch
import numpy as np
import scanpy as sc
from scipy.stats import wasserstein_distance


def gene_mean_corr(syndata: sc.AnnData, realdata: sc.AnnData):

    synmean = np.asarray(syndata.X.mean(axis=0)).ravel()
    realmean = np.asarray(realdata.X.mean(axis=0)).ravel()

    gene_corr = np.corrcoef(synmean, realmean)[0, 1]

    return gene_corr


def gene_variance_corr(syndata: sc.AnnData, realdata: sc.AnnData):

    synvar = np.asarray(syndata.X.var(axis=0)).ravel()
    realvar = np.asarray(realdata.X.var(axis=0)).ravel()

    gene_corr = np.corrcoef(synvar, realvar)[0, 1]

    return gene_corr


def libsize_corr(syndata: sc.AnnData, realdata: sc.AnnData):

    synlibsize =  np.asarray(syndata.X.sum(axis=1)).ravel()
    reallibsize =  np.asarray(realdata.X.sum(axis=1)).ravel()

    return wasserstein_distance(synlibsize, reallibsize)

    



    