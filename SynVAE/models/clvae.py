import torch
import torch.nn as nn
import torch.nn.functional as F

class CLVAE(nn.Module):

    def __init__(self, n_genes, latent_dim=30, hidden_dims=None, num_trials_labels: int = None, trials_embed_dim: int = None, num_cell_classes=None, cell_embed_dim:int = None):
        
        super().__init__()

        self.latent_dim = latent_dim
        self.n_genes = n_genes

        self.num_trials_labels = num_trials_labels
        self.num_cell_classes = num_cell_classes

        self.trials_embed_dim = trials_embed_dim
        self.cell_embed_dim = cell_embed_dim

        self.exp_embedding = None
        self.cell_embedding = None

        self.avg_exp_embedding = None
        self.avg_cell_embedding = None
        
        self.enc_input_dim = self.n_genes
        self.dec_input_dim =  self.latent_dim

        if hidden_dims is None:
            hidden_dims = [2000, 500, 250, 50]

        self.hidden_dims = hidden_dims

        if self.num_trials_labels is not None:
            self.dec_input_dim += self.trials_embed_dim
            self.exp_embedding = nn.Embedding(self.num_trials_labels, self.trials_embed_dim)

        if self.num_cell_classes is not None:
            self.dec_input_dim += self.cell_embed_dim
            self.enc_input_dim += self.cell_embed_dim
            self.cell_embedding = nn.Embedding(self.num_cell_classes, self.cell_embed_dim)
    
        self.dispersion = nn.Parameter(torch.ones(self.n_genes))

        self.encoder = self.build_encoder(self.enc_input_dim, self.hidden_dims)

        self.fn_mu, self.fn_logvar = self.build_latent(self.hidden_dims)

        self.decoder = self.build_decoder(self.n_genes, self.dec_input_dim, self.hidden_dims)

    def build_encoder(self, encoder_input_dim, hidden_dims):
        layers = []

        prev_dim = encoder_input_dim

        for dim in hidden_dims:

            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())

            prev_dim = dim

        return nn.Sequential(*layers)
       
    def build_decoder(self, n_genes, decoder_input_dim, hidden_dims):
        layers = []

        prev = decoder_input_dim

        for h in reversed(hidden_dims):

            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())

            prev = h

        layers.append(nn.Linear(prev, n_genes))

        return nn.Sequential(*layers)
    
    def build_latent(self, hidden_dims):

        if hidden_dims is None:
            last_hidden = 50
        else:
            last_hidden = hidden_dims[-1]

        mu = nn.Linear(last_hidden, self.latent_dim)
        logvar = nn.Linear(last_hidden, self.latent_dim)

        return mu, logvar

    def compute_avg_embedding(self):
        if self.exp_embedding is not None:
            self.avg_exp_embedding = self.exp_embedding.weight.mean(dim=0, keepdim=True)
        if self.cell_embedding is not None:
            self.avg_cell_embedding = self.cell_embedding.weight.mean(dim=0, keepdim=True)

    def concat_embedding(self, x, labels, embedding_layer):
        embeds = embedding_layer(labels)
        return torch.cat((x, embeds), dim=-1)
    
    def encode(self, x, cell_ids = None):

        if self.num_cell_classes is not None and cell_ids is not None:
            x = self.concat_embedding(x, cell_ids, self.cell_embedding)
        
        x = self.encoder(x)

        mu = self.fn_mu(x)

        logvar = self.fn_logvar(x)

        sigma = torch.exp(0.5 * logvar)

        noise = torch.randn_like(sigma)

        z = mu + sigma * noise

        return z, mu, logvar
    
    def decode(self, x, z, exp_ids=None, cell_ids=None):

        library_size = x.sum(dim=1, keepdim=True) # gets the total umi count of the cell

       # z, mu_z, logvar = self.forward_enc(x, cell_labels)

        if self.num_cell_classes is not None and cell_ids is not None:
            z = self.concat_embedding(z, cell_ids, self.cell_embedding)

        if self.num_trials_labels is not None and exp_ids is not None:
            z = self.concat_embedding(z, exp_ids, self.exp_embedding) # to support experimental biases

        probs = torch.softmax(self.decoder(z), dim=-1) # get the proportion of each gene in the lib size

        mu_x = library_size * probs

        theta = F.softplus(self.dispersion)

        #return z, mu_z, logvar, mu_x, theta
    
        return z, mu_x, theta
    
    def forward(self, x, exp_ids=None, cell_ids=None):
        z, mu_z, logvar = self.encode(x, cell_ids)
        z, mu_x, theta = self.decode(x, z, exp_ids, cell_ids)
        return z, mu_z, logvar, mu_x, theta
    