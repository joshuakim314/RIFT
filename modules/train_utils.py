import torch
import torch.nn.functional as F


def shifted_diff(seq_tensor, shift):
    res = seq_tensor - seq_tensor.roll(shift, 1)
    # zero out "future" times
    res[:, 0:shift] = 0
    return res


def shifted_diff_squared(seq_tensor, shift):
    res = seq_tensor - seq_tensor.roll(shift, 1)
    # zero out "future" times
    res[:, 0:shift] = 0
    return res**2


def shifted_diff_abs(seq_tensor, shift):
    assert shift >= 1, "Shift for shifted diff function must be 1 or above"
    res = seq_tensor - seq_tensor.roll(shift, 1)
    # zero out "future" times
    res[:, 0:shift] = 0
    return abs(res)


def ts_moving_average(seq_tensor, shift, mask=True):
    assert shift >= 1, "Shift for moving average function must be 1 or above"
    res = torch.cumsum(seq_tensor, 1)
    res[:, shift:] = res[:, shift:] - res[:, :-shift]
    res[:, shift - 1:] = res[:, shift - 1:] / shift
    if mask:
        res[:, 0:shift - 1] = 0
    return res


def ts_moving_var(seq_tensor, shift):
    assert shift >= 1, "Shift for moving average function must be 1 or above"
    seq_means = torch.mean(seq_tensor, 1)
    res = (seq_tensor.sub(seq_means.view(-1, 1)))**2
    res = ts_moving_average(res, shift, mask=False)
    res[:, 0:shift - 1] = 0
    return res


def ret_seq_indices(seq_tensor):
    """
    Returns the float indices of a torch tensor so the transformer network can attend to positions
    Closest positions have highest index (1); furthest positions have index 0.2
    """
    seq_len = seq_tensor.shape[1]
    batch_size = seq_tensor.shape[0]
    res = seq_tensor.clone()
    res[:, :] = (torch.arange(0, seq_len).repeat(
        batch_size).view(-1, seq_len).float() + 1 + seq_len // 4) / (5.0 / 4 * seq_len)
    return res


def seq_corr_1d(seq_tensor_x, seq_tensor_y):
    """
    Arguments
    ---------
    x : 1D torch.Tensor seq_len
    y : 1D torch.Tensor seq_len
    Returns
    scalar tensor of correlation
    -------

    """
    assert seq_tensor_x.shape == seq_tensor_y.shape, "Tensor shape mismatch for correlation function"
    seq_len = seq_tensor_x.shape[0]
    assert len(
        seq_tensor_x.shape) == 1, "Dimensionality 1 tensor expected for correlation function"
    mean_x = torch.mean(seq_tensor_x)
    mean_y = torch.mean(seq_tensor_y)
    x_diff = seq_tensor_x.sub(mean_x)
    y_diff = seq_tensor_y.sub(mean_y)
    # We use conv1d as a "hack" to compute correlation to take advantage of GPU acceleration; the hack is to pass the second time series as filter weights
    corr = F.conv1d(x_diff.view(1, -1, seq_len), y_diff.view(-1, 1, seq_len),
                    groups=1) / (torch.sum(x_diff**2) * torch.sum(y_diff**2)) ** 0.5
    corr[torch.isnan(corr)] = 0
    return corr.view(1)


def seq_corr_3d(seq_tensor_x, seq_tensor_y):
    """
    Arguments
    ---------
    x : 3D torch.Tensor batch_size * seq_len * embedding_dim
    y : 3D torch.Tensor batch_size * seq_len * embedding_dim
    Returns
    2D tensor of embedding correlations: batch_size * seq_len
    -------

    """
    assert seq_tensor_x.shape == seq_tensor_y.shape, "Tensor shape mismatch for correlation function"
    assert len(
        seq_tensor_x.shape) == 3, "Dimensionality 2 tensor expected for correlation function"
    batch_size = seq_tensor_x.shape[0]
    seq_len = seq_tensor_x.shape[1]
    embed_dim = seq_tensor_x.shape[2]
    res = seq_tensor_x[:, :, 0].clone()
    res[:, :] = 0
    for i in range(seq_len):
        mean_x = torch.mean(seq_tensor_x[:, i, :].view(
            batch_size, embed_dim), dim=1)
        mean_y = torch.mean(seq_tensor_x[:, i, :].view(
            batch_size, embed_dim), dim=1)
        x_diff = seq_tensor_x[:, i, :].view(
            batch_size, embed_dim).sub(mean_x.view(batch_size, 1))
        y_diff = seq_tensor_y[:, i, :].view(
            batch_size, embed_dim).sub(mean_y.view(batch_size, 1))
        corr = F.conv1d(x_diff.view(1, -1, embed_dim), y_diff.view(-1, 1, embed_dim), groups=batch_size).view(
            batch_size) / (torch.sum(x_diff**2, dim=1) * torch.sum(y_diff**2, dim=1)) ** 0.5
        res[:, i] = corr
    return res