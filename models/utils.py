import math
import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange

def exists(val):
    return val is not None

# decorators

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if not exists(mask):
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

# sampling helpers

def gumbel_noise(t, u=0, s=1):
    noise = torch.zeros_like(t).uniform_(u, s)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def gumbel_samples(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).max(dim = dim)

def gumbel_logits(t, temperature = 1., s=1, dim = -1, prob=False):

    probs = None
    logits = ((t / max(temperature, 1e-10)) + gumbel_noise(t, s=1.1))
    scores, index = logits.max(dim = dim)
    
    if prob:
        probs = F.softmax(logits, dim=-1).take(index).squeeze(0)  
    
    return scores, index, probs


def multinomial_logits(t, logits, temperature=1., k=1, dim=-1):

    # Sample from the filtered distribution
    probabilities = F.softmax(t.clone(), dim=-1)
    # print(f"probabilities: {probabilities}, probabilities.shape: {probabilities.shape}")
    token = torch.multinomial(probabilities.squeeze(0), 1)
    if token.dim() == 1:
        return logits.take(token), token

    value, index = logits.take(token).permute((1,0)), token.permute((1,0))
    print(f'value: {value}, index: {index}')
    return value, index


def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres=None, k=None, dim=-1, filter_value=float('-inf')):
    if thres:
        k = math.ceil((1 - thres) * logits.shape[-1])

    val, ind = torch.topk(logits, k, dim=dim)

    probs = torch.full_like(logits, filter_value)
    probs.scatter_(-1, ind, val)
    return probs


def _parameter_number(model):
    """
    统计模型参数量
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': f'{total_num/1e6:.3f} M', 'Trainable': f'{trainable_num/1e6:.3f} M'}


def param_head(model):
    """
    print head line of parameters for a model
    """

    it = 0
    for _, param in enumerate(model.parameters()):
        it+=1
        
        if it > 2:
            break
        return param