import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.conv1 = nn.Conv2d

x = np.random.rand(2, 3, 4)
print(x)
print(x[1, : , 2])