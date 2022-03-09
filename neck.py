import torch
from network_blocks import *
#delete
from backbone import *



class Neck(nn.Module):
    def __init__(self, scale=1):
        super(Neck, self).__init__()
        self.seq1 = CBL5(384//scale, 128//scale)
        self.seq2 = CBL5(768//scale, 256//scale)
        self.seq3 = CBL5(1024//scale, 512//scale)
        self.up1 = nn.Sequential(
            CBL(256//scale, 128//scale, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            CBL(512//scale, 256//scale, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, backbone_output):
        n3 = self.seq3(backbone_output[2])
        n2 = self.seq2(torch.cat([backbone_output[1], self.up2(n3)], dim=1))
        n1 = self.seq1(torch.cat([backbone_output[0], self.up1(n2)], dim=1))
        return [n1, n2, n3]

if __name__== "__main__":

    b1 = Backbone()
    b2 = Backbone(scale=2)
    n1 = Neck()
    n2 = Neck(scale=2)
    x = torch.randn(1,3,640,640)
    y = n1(b1(x))
    z = n2(b2(x))
    for i,j in zip(y,z):
        print(i.shape, j.shape)