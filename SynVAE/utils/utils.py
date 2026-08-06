import torch
from torch.nn import functional as F
from tqdm import tqdm
import scanpy as sc
from SynVAE.losses.vaeloss import get_kl_weight

def preprocessing(data):

    sc.pp.filter_cells(data, min_genes=1)

    return data

def sc_collate_fn(batch):
    x, cell_ids, exp_ids = zip(*batch)

    x = torch.stack(x)

    if cell_ids[0] is not None:
        cell_ids = torch.stack(cell_ids)
    else:
        cell_ids = None

    if exp_ids[0] is not None:
        exp_ids = torch.stack(exp_ids)
    else:
        exp_ids = None

    return x, cell_ids, exp_ids

def convert_onnx(model, input_tensor):
    model.eval()

    torch.onnx.export(model,
                      input_tensor,
                      "Model.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=False,
                      input_names=['modelInput'],
                      output_names=['modelOutput'],
                      dynamic_axes={'modelInput' :{0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}})

    print(" ")
    print('Model has been converted to ONNX')

def train_model(model, dataloader, optimizer, epochs, vae_loss_fn, kl_weight, enc_log1p, kl_annealing=False, early_stopping=True, min_delta = 1e-4, patience = 10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    losses = []
    rec_losses = []
    kl_losses = []

    best_loss =  float("inf")
    best_state = None

    pbar = tqdm(total=epochs)

    log_library = []

    for epoch in range(epochs):
        running_loss = 0.0
        running_rec_loss = 0.0
        running_kl_loss = 0.0
        n_batches = 0

        for batch, cell_id, exp_id in dataloader:
            log_library.append(torch.log(batch.sum(dim=1)))

            batch = batch.to(device)

            if cell_id is not None:
                cell_id = cell_id.to(device)

            if exp_id is not None:
                exp_id = exp_id.to(device)


            z, mu_z, logvar, mu_x, theta = model(batch, exp_id, cell_id, enc_log1p)

            if kl_annealing:
                kl_weight = get_kl_weight(epoch)
            else:
                kl_weight = kl_weight

            loss, rec_loss, kl_loss = vae_loss_fn(batch, mu_z, mu_x, theta, logvar, kl_weight, 1, 1e-6)

            if not torch.isfinite(loss):
                print(f"Loss became invalid at epoch {epoch}")
                break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_rec_loss += rec_loss.item()
            running_kl_loss += kl_loss.item()
            n_batches += 1

        losses.append(running_loss / n_batches)
        rec_losses.append(running_rec_loss / n_batches)
        kl_losses.append(running_kl_loss / n_batches)

        if early_stopping:
            if losses[-1] < best_loss - min_delta:
                best_loss = losses[-1]
                patience_counter = 0

                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        pbar.update(1)

    pbar.close()

    if best_state is not None and early_stopping:
        model.load_state_dict(best_state)
    
    log_lib = torch.cat(log_library)
    lib_mu = log_lib.mean()
    lib_std = log_lib.std()

    return losses, rec_losses, kl_losses, lib_mu, lib_std

def generate_synthetic_cells(model, n_cells, lib_mu, lib_std, cell_ids = None, exp_ids = None):

    device = next(model.parameters()).device
    model.eval()

    if cell_id is not None:
        cell_id = cell_id.to(device)
    if exp_id is not None:
        exp_id = exp_id.to(device)

    with torch.no_grad():

        z = torch.randn(n_cells, model.latent_dim, device=device)
        
        if model.cell_embedding is not None:

            if cell_ids is None:
                z = torch.cat((z, model.avg_cell_embedding.repeat(n_cells, 1)), dim=-1)
            else:
                z = torch.cat((z, model.cell_embedding(cell_ids)), dim=-1)

        if model.exp_embedding is not None:

            if exp_ids is None:
                z = torch.cat((z, model.avg_exp_embedding.repeat(n_cells, 1)), dim=-1)
            else:
                z = torch.cat((z, model.exp_embedding(exp_ids)), dim=1)

        library = torch.exp(torch.randn(n_cells, 1, device=z.device) * lib_std + lib_mu)

        # print(f"Library: {library[:10]}")

        # print(f"lib_mu: {lib_mu}")

        # print(f"lib_std: {lib_std}")


        mu = torch.clamp(library * torch.softmax(model.decoder(z), dim=-1), min=1e-8)
        theta = torch.clamp(F.softplus(model.dispersion), min=1e-8)


        logits = torch.log(mu) - torch.log(theta)

        nb = torch.distributions.NegativeBinomial(total_count=theta, logits=logits)

        counts = nb.sample()

    return counts.cpu()
