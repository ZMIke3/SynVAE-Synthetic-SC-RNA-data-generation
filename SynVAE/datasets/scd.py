import torch
from torch.utils.data import Dataset
import numpy as np
import scipy

class SingleCellDataset(Dataset):

    def __init__(self, adata, has_cell_labels:bool = False, has_trials_labels:bool = False, cells_key:str = None, trials_key:str = None):
        self.X = adata.X

        self.has_trials_labels = has_trials_labels
        self.has_cell_labels = has_cell_labels

        self.trials_labels_num = True

        self.cell_labels_to_id = None
        self.trials_labels_to_id = None

        self.len_cell_ids = None
        self.len_trials_ids = None

        if has_cell_labels and cells_key is None:
            raise ValueError("cells_key must be provided when has_cell_labels=True")

        if has_trials_labels and trials_key is None:
            raise ValueError("trials_key must be provided when has_trials_labels=True")
        
        if self.has_trials_labels:

            self.trials_labels = adata.obs[trials_key].to_numpy()

            self.len_trials_ids = len(self.trials_labels)

            if self.trials_labels.dtype.kind in ("U", "S", "O"):

                unique_trials = np.unique(self.trials_labels)

                self.trials_labels_to_id = {cell:i for i, cell in enumerate(unique_trials)}

                self.trials_labels_num = False

        if self.has_cell_labels:

            self.cell_labels = adata.obs[cells_key].to_numpy()

            unique_cells = np.unique(self.cell_labels)

            self.cell_labels_to_id = {cell:i for i, cell in enumerate(unique_cells)}

            self.len_cell_ids = len(self.cell_labels_to_id)

    def get_label_to_id_dicts(self):
        return self.cell_labels_to_id, self.trials_labels_to_id

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        x = self.X[idx]

        cell_id = None
        trials_id = None

        if scipy.sparse.issparse(x):
            x = x.toarray().squeeze()

        if self.has_cell_labels:
            cell_id = torch.tensor(self.cell_labels_to_id[self.cell_labels[idx]], dtype=torch.int64)

        if self.has_trials_labels:
            if self.trials_labels_num:
                trials_id = torch.tensor(self.trials_labels[idx], dtype=torch.int64)
            else:
                trials_id = torch.tensor(self.trials_labels_to_id[self.trials_labels[idx]], dtype=torch.int64)
        
        return torch.tensor(x, dtype=torch.float32), cell_id, trials_id
