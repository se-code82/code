# From https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py with 
# some modifications

import torch
from torch import nn
from torch.nn import init


class CA(nn.Module):
    def __init__(self, channels):
        super(CA, self).__init__()

        self.conv7 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        # self.relu7 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x7 = self.conv5(x)
        x7 = self.relu5(x7)

        x5 = self.conv5(x)
        x5 = self.relu5(x5)

        x3 = self.conv3(x)
        x3 = self.relu3(x3)


        out1 = x3 + x5
        out2 = x7 + x5
        out3 = x3 + x7
        out4 = out1 + out2 + out3

        out6 = self.conv1(out4)
        out = out4 * x3
        out = out + x + out6
        out = self.softmax(out)

        return out
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
##################################################################
        self.ca = CA(channels=channel)
###################################################################
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x) :
##########################################################
        x = self.ca(x)
############################################################
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x) :
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result],1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, input_dim, reduction, input_size, out_dim):
        super().__init__()
        self.input_size = input_size
        self.ca = ChannelAttention(channel=input_dim, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=1)

        self.proj = nn.Linear(input_dim, out_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.input_size ** 2
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, self.input_size, self.input_size)
        
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        out = out + residual
        out = out.view(B, C, L).permute(0, 2, 1).contiguous()
        return self.proj(out)
        


    