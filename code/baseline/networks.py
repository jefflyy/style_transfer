import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out 

# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.AdaIN = AdaIN()

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True, track_running_stats=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True, track_running_stats=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2 )
        self.in3_d = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2 )
        self.in2_d = nn.InstanceNorm2d(32, affine=True, track_running_stats=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True, track_running_stats=True)

    def forward(self, x):
        # encode
        y1 = self.relu(self.in1_e(self.conv1(x)))
        y2 = self.relu(self.in2_e(self.conv2(y1)))
        y3 = self.relu(self.in3_e(self.conv3(y2)))

        # residual layers
        y = self.res1(y3)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        #y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y


# Features of VGG16
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

# Features of ResNet
class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        rn = models.resnext50_32x4d(weights='IMAGENET1K_V2')
        self.layer0 = nn.Sequential(
            rn.conv1,
            rn.bn1,
            rn.relu,
            rn.maxpool
        )
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, x):
        p0 = self.layer0(x)
        p1 = self.layer1(p0)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        return (p0, p1, p2, p3, p4)

# AdaIN
class AdaIN(nn.Module):
    def __init__(self) -> None:
        super(AdaIN, self).__init__()

    def forward(self, content, style, alpha = 1.0):
        eps = 1e-5
        # print(content.size())
        b, c, h, w = content.size()

        content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim = 2, keepdim = True)
        style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim = 2, keepdim = True)

        normalized_content = (content.view(b, c, -1) - content_mean) / (content_std + eps)
        stylized_content = (normalized_content * style_std) + style_mean

        out = (1 - alpha) * content + alpha * stylized_content.view(b, c, h, w)
        return out


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.adain = AdaIN()
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # print(input.size())
        # print(target.size())
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0, eval = False):
        # print(content.size())
        # print(style.size())
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = self.adain(content_feat, style_feats[-1])
        # t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        if eval:
            return g_t
        else:
            g_t_feats = self.encode_with_intermediate(g_t)

            loss_c = self.calc_content_loss(g_t_feats[-1], t)
            loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
            for i in range(1, 4):
                loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
            return loss_c, loss_s

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


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
        out = self.reflection_pad(out)
        out = F.relu6(self.in2(self.conv2(out)))
        out = self.in3(self.conv3(out))
        if self.residual:
            out = x + out
        return out


class UpsampleConv(nn.Module):
    def __init__(self, inp_c, out_c, kernel_size, stride, upsample=2):
        super().__init__()
        if upsample:
            self.upsample = nn.Upsample(mode='nearest', scale_factor=upsample)
        else:
            self.upsample = None
        self.conv1 = Bottleneck(inp_c, out_c, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample is not None:
            x_in = self.upsample(x_in)
        out = F.relu(self.conv1(x_in))
        return out


class TransformerMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv Layers
        self.reflection_pad = nn.ReflectionPad2d(9//2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, bias=False)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = Bottleneck(32, 64, kernel_size=3, stride=2)
        self.conv3 = Bottleneck(64, 128, kernel_size=3, stride=2)
        # Residual Layers
        self.res1 = Bottleneck(128, 128,  3, 1)
        self.res2 = Bottleneck(128, 128,  3, 1)
        self.res3 = Bottleneck(128, 128,  3, 1)
        self.res4 = Bottleneck(128, 128,  3, 1)
        self.res5 = Bottleneck(128, 128,  3, 1)
        # Upsampling Layers
        self.upconv1 = UpsampleConv(128, 64, kernel_size=3, stride=1)
        self.upconv2 = UpsampleConv(64, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = F.relu(self.in1(self.conv1(out)))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.conv4(self.reflection_pad(out))
        return out

