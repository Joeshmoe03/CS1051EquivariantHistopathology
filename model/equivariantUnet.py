import torch
from model.equivariant import ScalarToScalar, Downsample, Upsample, ScalarSigmoid

class EqUnetVariant(torch.nn.Module):
    def __init__(self, img_channels=3, in_channels=64, out_channels=3, k=3):
        super().__init__()
        
        # downsampling
        self.img_channels = img_channels
        p = 1
        in_channels = in_channels
        self.out_channels = out_channels

        self.c1 = ScalarToScalar(in_channels = self.img_channels, out_channels=in_channels, kernel_size=k, padding=p)
        self.p1 = Downsample()
        self.s1 = ScalarSigmoid()

        self.c2 = ScalarToScalar(in_channels = in_channels, out_channels=in_channels*2, kernel_size=k, padding=p)
        self.p2 = Downsample()
        self.s2 = ScalarSigmoid()

        self.c3 = ScalarToScalar(in_channels = in_channels*2, out_channels=in_channels*4, kernel_size=k, padding=p)
        self.p3 = Downsample()
        self.s3 = ScalarSigmoid()

        self.c4 = ScalarToScalar(in_channels = in_channels*4, out_channels=in_channels*8, kernel_size=k, padding=p)
        self.p4 = Downsample()
        self.s4 = ScalarSigmoid()

        self.c5 = ScalarToScalar(in_channels = in_channels*8, out_channels=in_channels*16, kernel_size=k, padding=p)
        self.p5 = Downsample()
        self.s5 = ScalarSigmoid()
        
        # upsampling
        self.p5_ = Upsample()
        self.c5_ = ScalarToScalar(in_channels = in_channels*16+in_channels*8, out_channels=in_channels*8, kernel_size=k, padding=p)
        self.s5_ = ScalarSigmoid()

        self.p4_ = Upsample()
        self.c4_ = ScalarToScalar(in_channels = in_channels*8+in_channels*4, out_channels=in_channels*4, kernel_size=k, padding=p)
        self.s4_ = ScalarSigmoid()
        
        self.p3_ = Upsample()
        self.c3_ = ScalarToScalar(in_channels = in_channels*4+in_channels*2, out_channels=in_channels*2, kernel_size=k, padding=p)
        self.s3_ = ScalarSigmoid()

        self.p2_ = Upsample()
        self.c2_ = ScalarToScalar(in_channels = in_channels*2+in_channels, out_channels=in_channels, kernel_size=k, padding=p)
        self.s2_ = ScalarSigmoid()
        
        self.p1_ = Upsample()
        self.c1_ = ScalarToScalar(in_channels = in_channels+self.img_channels, out_channels=self.out_channels, kernel_size=k, padding=p)
           
    def forward(self,x):
        x0 = [x] # 64x64
        x = self.s1(self.p1(self.c1(x)))
        x0.append(x) # 32x32
        x = self.s2(self.p2(self.c2(x)))
        x0.append(x) # 16x16
        x = self.s3(self.p3(self.c3(x)))
        x0.append(x) # 8x8
        x = self.s4(self.p4(self.c4(x)))
        x0.append(x) # 4x4
        x = self.s5(self.p5(self.c5(x)))
        # 2x2
        
        # now upsampling
        x = self.p5_(x) # upsample        
        # # concat
        x = torch.concatenate((x,x0.pop()),-3)        
        # # conv
        x = self.c5_(x)
        x = self.s5_(x)
        
        x = self.p4_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c4_(x)
        x = self.s4_(x)
        
        x = self.p3_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c3_(x)
        x = self.s3_(x)
        
        x = self.p2_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c2_(x)
        x = self.s2_(x)
        
        x = self.p1_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c1_(x)
        
        return x