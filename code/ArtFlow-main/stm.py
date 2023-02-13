import torch
from torch import nn
import torch.nn.functional as F

from function import calc_mean_std


class AdaIN(nn.Module):
    def __init__(self, in_c=-1):
        super().__init__()
        
    def forward(self, content, style):
        # print(f'adain {content.shape} {style.shape}')

        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Bottleneck(nn.Module):
    def __init__(self, inp_c, out_c, kernel_size, stride, t=1):
        assert stride in [1, 2], 'stride must be either 1 or 2'
        super().__init__()
        self.residual = stride == 1 and inp_c == out_c
        pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv1 = nn.Conv2d(inp_c, t*inp_c, 1, 1, bias=False)
        self.in1 = nn.InstanceNorm2d(t*inp_c, affine=True)
        self.conv2 = nn.Conv2d(t*inp_c, t*inp_c, kernel_size, stride,
                               groups=t*inp_c, bias=False)
        self.in2 = nn.InstanceNorm2d(t*inp_c, affine=True)
        self.conv3 = nn.Conv2d(t*inp_c, out_c, 1, 1, bias=False)
        self.in3 = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        out = F.relu6(self.in1(self.conv1(x)))
        out = F.relu6(self.in2(self.conv2(out)))
        out = self.reflection_pad(out)
        out = self.in3(self.conv3(out))
        if self.residual:
            out = x + out
        return out

class RBMAD(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c_rb = Bottleneck(in_c, in_c, 3, 1)
        self.s_rb = Bottleneck(in_c, in_c, 3, 1)
        self.conv_c = nn.Conv2d(in_c, in_c // 2, 1)
        self.conv_s = nn.Conv2d(in_c, in_c // 2, 1)
        self.adain = AdaIN()
    
    def forward(self, content, style):
        c_rb = self.c_rb(content)
        s_rb = self.s_rb(content)
        conv_c = self.conv_c(c_rb)
        conv_s = self.conv_s(s_rb)
        scm = torch.cat([conv_c, conv_s], dim=1)
        return self.adain(content, scm)
