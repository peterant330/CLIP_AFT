import torch
from train.utils import project_perturbation, normalize_grad


def pgd(
        forward,
        loss_fn,
        data_clean,
        targets,
        norm,
        eps,
        iterations,
        stepsize,
        output_normalize,
        eps2 = 0,
        perturbation=None,
        mode='min',
        momentum=0.9,
        verbose=False
):
    """
    Minimize or maximize given loss
    """
    # make sure data is in image space
    assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6

    if perturbation is None:
        perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)
    for i in range(iterations):
        noise = torch.randn_like(data_clean) / 255
        data_noisy = (data_clean + noise).clamp(0, 1)
        perturbation.requires_grad = True
        with torch.enable_grad():
            out = forward(data_noisy + perturbation, output_normalize=output_normalize)
            loss = loss_fn(out, targets)
            if norm == "elastic" and eps2 > 0:
                loss = loss - 0.25/eps2 * torch.square(torch.abs(perturbation)-eps).clamp(0).sum(dim=[1,2,3]).mean()
            elif norm == "huber" and eps2 > 0:
                loss = loss - eps2 / eps * torch.linalg.norm(perturbation, dim=1).mean()
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            gradient = gradient
            if gradient.isnan().any():  #
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                gradient[gradient.isnan()] = 0.
            # normalize
            gradient = normalize_grad(gradient, p=norm)
            # momentum
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)
            # update
            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            # project
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(
                data_clean + perturbation, 0, 1
            ) - data_clean  # clamp to image space
            assert not perturbation.isnan().any()
            assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
                data_clean + perturbation
            ) > -1e-6

            # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
    # todo return best perturbation
    # problem is that model currently does not output expanded loss
    return data_clean + perturbation.detach()
