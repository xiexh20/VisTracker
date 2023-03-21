import torch


def clips2seq_fast(clips, step, window_size):
    "10 times faster version"
    assert step == 1, 'currently only support step size 1!'
    B, T, D = clips.shape
    L = (B-1)*step + window_size
    out_all = torch.zeros(L, window_size//step, D).to(clips.device)

    masks = []
    for t in range(T):
        out_b_idx = torch.arange(0, L).to(clips.device)
        in_b_idx = torch.arange(-T+1+t, -T+L+t+1).to(clips.device)
        in_t_idx = T - 1 - t
        mask = (in_b_idx < B) & (in_b_idx >=0)
        out_all[out_b_idx[mask], t] = clips[in_b_idx[mask], in_t_idx]
        masks.append(mask)
    masks = torch.stack(masks, 1)
    seq = torch.sum(out_all, 1) / torch.sum(masks, 1).unsqueeze(-1)
    return seq


def slide_window_to_sequence(slide_window,window_step,window_size):
    """

    Args:
        slide_window: denoised data, (B, T, D)
        window_step: distance between starts of two clips (can overlap)
        window_size: T
    step:1, window size: 32, pose shape: torch.Size([1151, 32, 42])
    Returns: (seq_len, D)

    """
    if window_step == 1:
        seq = clips2seq_fast(slide_window, window_step, window_size)
        return seq

    # old version
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0) # take the mean of all data!

    sequence = torch.stack(sequence)
    return sequence