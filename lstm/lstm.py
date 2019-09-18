import torch.nn as nn
import torch as t
import torchvision as tv
from torch.autograd import variable


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)

        )


lstm = nn.LSTM(input_size=4,
               hidden_size=10,
               batch_first=True)
print(lstm)
