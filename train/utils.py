import sys
import wandb
from time import sleep
import os
import torch
import open_clip
from torchvision.transforms import transforms

def init_wandb(project_name, model_name, config, **wandb_kwargs):
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    while True:
        try:
            wandb_run = wandb.init(
                project=project_name, name=model_name, save_code=True,
                config=config, **wandb_kwargs,
                )
            break
        except Exception as e:
            print('wandb connection error', file=sys.stderr)
            print(f'error: {e}', file=sys.stderr)
            sleep(1)
            print('retrying..', file=sys.stderr)
    return wandb_run

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, projector):
        super().__init__()
        self.model = model
        self.projector = projector

    def forward(self, vision):
        embedding = self.model(vision)
        embedding = self.projector(embedding)
        return embedding

def load_clip_model(clip_model_name, pretrained, beta=0.):
    model_name = clip_model_name
    try:  # try loading only visual model
        model, image_processor = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained='openai', device='cpu'
        )
        if pretrained != 'openai':
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = pretrained
            # if beta non-zero interpolate between clean and pretrained ckpts

            if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            else:
                model.visual.load_state_dict(checkpoint)
    except RuntimeError as e:  # try loading whole model
        print(f'error: {e}', file=sys.stderr)
        print('retrying by loading whole model..', file=sys.stderr)
        torch.cuda.empty_cache()
        model, _, image_processor = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained, force_quick_gelu=True, device='cpu'
        )
    model.eval()

    # Remove the Normalize transform by creating a new Compose object
    preprocessor_no_norm = transforms.Compose(image_processor.transforms[:-1])
    normalizer = image_processor.transforms[-1]
    return model, preprocessor_no_norm, normalizer

import torch.nn.functional as F
def project_perturbation(perturbation, eps, norm):
    if norm in ['inf', 'linf', 'Linf', 'huber']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif norm in ['group']:
        delta_square = perturbation ** 2
        grad_norm = delta_square.reshape(-1, 3, 16, 14, 16, 14).permute(0, 1, 2, 4, 3, 5) \
            .flatten(4, 5).sum(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 14, 14)
        grad_norm = torch.sqrt(grad_norm).permute(0, 1, 2, 4, 3, 5).reshape(-1, 3, 224, 224)
        ratio = torch.where(grad_norm > eps, eps / grad_norm, 1.)
        pert_normalized = perturbation * ratio.detach()
        return pert_normalized
    elif norm in ["elastic"]:
        return perturbation
    elif norm in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError(f'Norm {norm} not supported')



def normalize_grad(grad, p):
    if p in ['inf', 'linf', 'Linf', "huber"]:
        return grad.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2', "elastic"]:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)
    elif p in ["group"]:
        grad_norm = torch.square(grad).reshape(-1, 3, 16, 14, 16, 14).permute(0, 1, 2, 4, 3, 5) \
            .flatten(4, 5).sum(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 14, 14)
        grad_norm = torch.sqrt(grad_norm).permute(0, 1, 2, 4, 3, 5).reshape(-1, 3, 224, 224)
        return grad / grad_norm
