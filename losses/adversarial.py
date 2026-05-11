import torch
import torch.autograd as autograd


def gradient_penalty_wgan(
    D: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    mask: torch.Tensor,
    gp_weight: float = 10.0,
) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    interp = eps * real + (1.0 - eps) * fake
    interp = interp.detach().requires_grad_(True)
    d_interp = D(interp, mask)
    grad = autograd.grad(
        outputs=d_interp.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp_weight * gp


def wasserstein_disc_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return -(d_real.mean() - d_fake.mean())


def wasserstein_gen_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()
