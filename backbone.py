import torch

from network_blocks import *


class Backbone(nn.Module):
    def __init__(self, scale=1):
        super(Backbone, self).__init__()
        self.seq1 = nn.Sequential(
            CBL(3, 32//scale, 3, 1, 1),
            ResX(1, 32//scale, 64//scale, 3, 2, 1),
            ResX(2, 64//scale, 128//scale, 3, 2, 1),
            ResX(5, 128//scale, 256//scale, 3, 2, 1)
        )
        self.seq2 = ResX(5, 256//scale, 512//scale, 3, 2, 1)
        self.seq3 = ResX(2, 512//scale, 1024//scale, 3, 2, 1)

    def forward(self, x):
        b1 = self.seq1(x)
        b2 = self.seq2(b1)
        b3 = self.seq3(b2)
        return [b1, b2, b3]

if __name__== "__main__":
    b1 = Backbone()
    b2 = Backbone(scale=2)
    x = torch.randn(1,3,320,320)
    y = b1(x)
    z = b2(x)
    for i,j in zip(y,z):
        print(i.shape, j.shape)