import torch.nn as nn
import torch

# n -> batch size, l -> sequence length, c -> feature dim, d -> output dim
def do_lstm(n, l, c, d):
    lstm = nn.LSTM(c, d//2, num_layers=2, bidirectional=True)
    given = torch.randn(n, l, c)
    hidden = (torch.randn(4, l, d//2), torch.randn(4, l, d//2))
    output, hidden = lstm(given, hidden)
    return output

print(do_lstm(4, 5, 5, 10))