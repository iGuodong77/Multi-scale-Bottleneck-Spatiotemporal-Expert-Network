def channel_normalization(tensor, eps=1e-8):
    # tensor shape: [C, H, W]
    min_vals = tensor.view(tensor.shape[0], -1).min(dim=1).values.view(-1, 1, 1)
    max_vals = tensor.view(tensor.shape[0], -1).max(dim=1).values.view(-1, 1, 1)
    return (tensor - min_vals) / (max_vals - min_vals + eps)