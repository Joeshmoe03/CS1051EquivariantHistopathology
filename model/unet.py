import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    """MaxPooling for downsampling"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.pool(x)

class Upsample(nn.Module):
    """Nearest-neighbor upsampling (NO TRANSPOSE CONVOLUTION SADLY BCUZ ITS NOT AVAILABLE IN EQUIVARIANT VERSION, TO MAKE IT FAIR)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

class SingleConv(nn.Module):
    """A single convolution layer (analogous to ScalarToScalar)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
    
    def forward(self, x):
        return self.conv(x)

class UnetVariant(nn.Module):
    def __init__(self, img_channels=3, out_channels=3, in_channels=64):
        super().__init__()
        self.img_channels = img_channels
        self.out_channels = out_channels
        
        # Downsampling path
        self.c1 = SingleConv(img_channels, in_channels)
        self.p1 = Downsample()
        self.s1 = nn.ReLU()

        self.c2 = SingleConv(in_channels, in_channels * 2)
        self.p2 = Downsample()
        self.s2 = nn.ReLU()

        self.c3 = SingleConv(in_channels * 2, in_channels * 4)
        self.p3 = Downsample()
        self.s3 = nn.ReLU()

        self.c4 = SingleConv(in_channels * 4, in_channels * 8)
        self.p4 = Downsample()
        self.s4 = nn.ReLU()

        self.c5 = SingleConv(in_channels * 8, in_channels * 16)
        self.p5 = Downsample()
        self.s5 = nn.ReLU()

        # upsampling
        self.p5_ = Upsample()
        self.c5_ = SingleConv(in_channels * 16 + in_channels * 8, in_channels * 8)
        self.s5_ = nn.ReLU()

        self.p4_ = Upsample()
        self.c4_ = SingleConv(in_channels * 8 + in_channels * 4, in_channels * 4)
        self.s4_ = nn.ReLU()

        self.p3_ = Upsample()
        self.c3_ = SingleConv(in_channels * 4 + in_channels * 2, in_channels * 2)
        self.s3_ = nn.ReLU()

        self.p2_ = Upsample()
        self.c2_ = SingleConv(in_channels * 2 + in_channels, in_channels)
        self.s2_ = nn.ReLU()

        self.p1_ = Upsample()
        self.c1_ = SingleConv(in_channels + img_channels, out_channels)
           
    def forward(self, x):
        x0 = [x]  # Skip cons

        # Downsample
        x = self.s1(self.p1(self.c1(x)))
        x0.append(x)
        x = self.s2(self.p2(self.c2(x)))
        x0.append(x)
        x = self.s3(self.p3(self.c3(x)))
        x0.append(x)
        x = self.s4(self.p4(self.c4(x)))
        x0.append(x)
        x = self.s5(self.p5(self.c5(x)))
        
        # Upsampling
        x = self.p5_(x)     
        x = torch.cat([x, x0.pop()], dim=1)   
        x = self.s5_(self.c5_(x))

        x = self.p4_(x)     
        x = torch.cat([x, x0.pop()], dim=1)   
        x = self.s4_(self.c4_(x))

        x = self.p3_(x)     
        x = torch.cat([x, x0.pop()], dim=1)   
        x = self.s3_(self.c3_(x))

        x = self.p2_(x)     
        x = torch.cat([x, x0.pop()], dim=1)   
        x = self.s2_(self.c2_(x))

        x = self.p1_(x)     
        x = torch.cat([x, x0.pop()], dim=1)   
        x = self.c1_(x)  # Final layer (no activation)
        
        return x