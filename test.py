import scanpy as sc
from SynVAE.synvae.synvae import SynVae
from SynVAE.utils.utils import  preprocessing
from torch.utils.data import DataLoader

def main():

    # Used the following data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2895282
    # Utilised Cell Typist to get lables for the dataset "Immune Low"

    data = sc.read_h5ad("train_data.h5ad")

    data = preprocessing(data)

    model = SynVae(data)

    model.fit(num_workers=0)

    real_loader = DataLoader(model.dataset, batch_size=120, shuffle=False, collate_fn=model.get_collate_fn())
    model.plotlatent(real_loader)

    n_cells = 5000

    samples = model.sample(n_cells)

    model.umap()

    model.saveSynData("syndata.h5ad")

    model.save("test_model.synvae")


if __name__ == "__main__":
    main()