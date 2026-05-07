import torch

def huber_loss(model_pred, target, delta=1.0, reduce=True):
    # 计算误差
    error = model_pred - target
    
    # 如果误差的绝对值小于 delta，使用 L2 损失，否则使用 L1 损失
    if reduce:
        loss = torch.mean(
            torch.sqrt(error ** 2 + delta ** 2) - delta
        )
    else:
        loss = torch.sqrt(error ** 2 + delta ** 2) - delta
    
    return loss

def gaussian_mixture(k, locs, var, mode_probs=None):
    if mode_probs is None:
        mode_probs = [1 / len(locs)] * len(locs)

    def _gaussian(x):
        prob = [
            mode_probs[i] * torch.exp(-torch.tensor([(x - loc) ** 2 / var]))
            for i, loc in enumerate(locs)
        ]
        # prob.append(mode_prob * torch.exp(-torch.tensor([(x) ** 2 / var])))
        return sum(prob)

    return _gaussian