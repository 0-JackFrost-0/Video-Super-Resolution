<<<<<<< HEAD
# Contains the neural network
# Task:- Implement the Neural Network module according to problem statement specifications


from torch import nn


class Net(nn.Module):
    def __init__(self, base_filter, num_channels=1):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9,  stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

=======
# Contains the neural network
# Task:- Implement the Neural Network module according to problem statement specifications


from torch import nn


class Net(nn.Module):
    def __init__(self, base_filter, num_channels=1):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9,  stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

>>>>>>> 94a66407832858375f2f105b3f8d2e54882416ce
