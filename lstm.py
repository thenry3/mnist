import torch.nn as nn
import torch


def do_lstm(n, l, c, d):
    """
    n -> batch size, l -> sequence length, c -> feature dim, d -> output dim
    """
    lstm = nn.LSTM(c, d//2, num_layers=2, bidirectional=True, batch_first=True)
    given = torch.randn(n, l, c)
    hidden = (torch.randn(4, n, d//2), torch.randn(4, n, d//2))
    output, hidden = lstm(given, hidden)
    return output


def lstm(n, l, c, d):
    lstm = nn.LSTM(c, d//2, num_layers=2, bidirectional=True)
    given = torch.randn(l, n, c)
    hidden = (torch.randn(4, n, d//2), torch.randn(4, n, d//2))
    output, hidden = lstm(given, hidden)
    print(output)
    return torch.stack([output[:, i] for i in range(n)])


# print(do_lstm(4, 5, 5, 10))
print(lstm(4, 5, 5, 10))
