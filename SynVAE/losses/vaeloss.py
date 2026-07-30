import torch


def VAELoss(x, mu_z, mu_x, theta, log_var, kl_weight=0.5, rec_weight=1, eps = 1e-6,):

    mu_x = torch.clamp(mu_x, min=1e-8)

    rec_loss = -(
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * torch.log(theta / (theta + mu_x + eps))
        + x * torch.log(mu_x / (theta + mu_x + eps))
    ).sum(dim=-1).mean()

    kl = (1 + log_var - mu_z**2 - torch.exp(log_var)).flatten(1)

    kl_loss = -0.5 * torch.sum(kl, dim=-1)

    kl_loss = torch.mean(kl_loss)

    return rec_weight * rec_loss + kl_weight * kl_loss, rec_loss, kl_loss

def get_kl_weight(epoch, warmup_epochs=25, target=0.01):
    return target * min(1.0, epoch / warmup_epochs)
