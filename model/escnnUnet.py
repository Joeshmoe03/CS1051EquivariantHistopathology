#TODO: FIX BROKEN CONCATS

import torch
from torch import nn
import e2cnn.gspaces as gspaces
import e2cnn.nn as enn

class escnnUnetVariant(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(escnnUnetVariant, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        # r2 actions specified as the set of rotations in r2
        self.r2_act = gspaces.Rot2dOnR2(N=4)

        self.in_type = enn.FieldType(self.r2_act, self.in_channels * [self.r2_act.trivial_repr])
        self.out_type = enn.FieldType(self.r2_act, 64 * [self.r2_act.trivial_repr])

        # DOUBLE CONV BLOCK
        self.inblock1 = enn.SequentialModule(
            enn.R2Conv(in_type = self.in_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.inblock2 = enn.SequentialModule(
            enn.R2Conv(in_type = self.in_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )

        # MAX POOL THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.down1_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.down1_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )

        # MAX POOL THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 256 * [self.r2_act.trivial_repr])
        self.down2_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.down2_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )

        # MAX POOL THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 512 * [self.r2_act.trivial_repr])
        self.down3_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.down3_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )

        # MAX POOL THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 1024 * [self.r2_act.trivial_repr])
        self.down4_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.down4_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )

        # UPSAMPLE THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 512 * [self.r2_act.trivial_repr])
        self.up1_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.up1_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        # UPSAMPLE THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 256 * [self.r2_act.trivial_repr])
        self.up2_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.up2_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        # UPSAMPLE THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.up3_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.up3_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        # UPSAMPLE THEN DOUBLE CONV BLOCK
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, 64 * [self.r2_act.trivial_repr])
        self.up4_1 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )
        self.up4_2 = enn.SequentialModule(
            enn.PointwiseMaxPool(self.out_type, kernel_size=2, stride=2),
            enn.R2Conv(in_type = self.out_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type)
        )

        # FINAL CONV
        self.in_type = self.out_type
        self.out_type = enn.FieldType(self.r2_act, self.out_channels * [self.r2_act.trivial_repr])
        self.finalconv = enn.R2Conv(in_type = self.in_type, out_type = self.out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False)

    def forward(self, x):
        x1 = self.inblock1(x)
        x2 = self.down1_1(x1)
        x3 = self.down2_1(x2)
        x4 = self.down3_1(x3)
        x5 = self.down4_1(x4)

        y5 = self.down4_2(x5)
        y4 = self.up1_1(y5) + x4
        y3 = self.up2_1(y4) + x3
        y2 = self.up3_1(y3) + x2
        y1 = self.up4_1(y2) + x1
        y0 = self.finalconv(self.up4_2(y1))
        return y0

eqModel = escnnUnetVariant(in_channels=3, out_channels=3)

x = torch.randn(1, 3, 256, 256)
r2_act = gspaces.Rot2dOnR2(N=4)
input_type = enn.FieldType(r2_act, 3 * [r2_act.trivial_repr])
x_geom = enn.GeometricTensor(x, input_type)

print("Expected input type:", eqModel.inblock1.in_type)
print("Provided input type:", x_geom.type)

# forward pass
y = eqModel(x_geom)