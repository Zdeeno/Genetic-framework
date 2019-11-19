import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def visualise(chromosome, timeseries, price_ts, filter_per_ts, savefig, device=None):
    """
    :param chromosome: (length)
    :param timeseries: (data, N)
    :return: percent earned per chromosome
    """
    torch.no_grad()  # setup torch
    width = timeseries.shape[1] * filter_per_ts
    filter_rows = width
    filter_len = int(chromosome.shape[0]/width)
    conv_filter = np.resize(np.transpose(chromosome), (filter_rows, 1, filter_len))
    torch_filter = torch.nn.Conv1d(1, filter_per_ts, conv_filter.shape[2], bias=False)

    outputs = []
    for i in range(int(width/filter_per_ts)):
        np_ts = timeseries[:-1, i]  # I dont really care about last action
        torch_ts = torch.from_numpy(np.resize(np_ts, (1, 1, np_ts.size)))
        start = i*filter_per_ts
        idxs = []
        for j in range(i*filter_per_ts, filter_rows, width):
            for k in range(filter_per_ts):
                idxs.append(j + k)
        torch_filter.weight = torch.nn.Parameter(torch.from_numpy(conv_filter[idxs, :, :]), requires_grad=False)
        if device is not None:
            torch_filter.cuda(device)
            torch_ts.cuda(device)
        out = torch_filter(torch_ts)
        outputs.append(out)

    # sum with other timeseries
    out_sum = outputs[0]
    for i in range(1, len(outputs)):
        out_sum += outputs[i]

    # sum with other filters
    final_sum = out_sum[:, 0::filter_per_ts, :]
    for i in range(1, filter_per_ts):
        final_sum += out_sum[:, i::filter_per_ts, :]

    # find buy and sell signals
    buys = (final_sum > 1).double()
    sells = (final_sum < 1).double()
    actions = buys - sells
    actions_idx = torch.nonzero(actions, as_tuple=True)

    # to cpu and cast to another type
    if device is not None:
        actions.to(torch.device("cpu"))
        actions_idx.to(torch.device("cpu"))

    actions = actions[0].numpy()
    bias = np.mean(actions)
    actions_idx = np.asarray(actions_idx[2])

    plt.plot(price_ts[filter_len:], lw=0.15)

    last_action = 0
    for i in range(actions_idx.size):
        idx = actions_idx[i]
        action = actions[0, idx]
        if action != last_action:
            if action == 1:
                plt.axvline(idx, -100, 100, color="g", lw=0.1)
            else:
                plt.axvline(idx, -100, 100, color="r", ls="--", lw=0.1)
            last_action = action

    if savefig:
        i = 0
        filename = "chart"
        while os.path.exists('{}{:d}.eps'.format(filename, i)):
            i += 1
        plt.savefig('{}{:d}.eps'.format(filename, i), format="eps")
        plt.close()
    else:
        plt.show()

    return bias
