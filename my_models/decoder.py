import torch
from torch import Tensor
from torch import nn
from typing import Optional
from hparams import hparams

class Decoder(nn.Module):
    def __init__(self, feature_channels, audio_decoder_channels):
        super().__init__()
        self.conv_4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain_4 = AdaIN(feature_channels[3]*2,audio_decoder_channels[3])
        
        self.pixel_suffle_3 = nn.PixelShuffle(upscale_factor=2)
        self.adain_3 = AdaIN(feature_channels[2]*2,audio_decoder_channels[2])
        
        self.pixel_suffle_2 = nn.PixelShuffle(upscale_factor=2)
        self.adain_2 = AdaIN(feature_channels[1]*2,audio_decoder_channels[1])
        
        self.pixel_suffle_1 = nn.PixelShuffle(upscale_factor=2)
        self.adain_1 = AdaIN(feature_channels[0]*2,audio_decoder_channels[0])
        
        self.pixel_suffle_0 = nn.PixelShuffle(upscale_factor=2)
        self.adain_0 = AdaIN(8+6,1*80*16)
        
        # Final layer
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=14, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.Sigmoid=torch.nn.Sigmoid()
        

    def forward(self, f0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                 a0: Tensor, a1: Tensor, a2: Tensor, a3: Tensor, a4: Tensor):
        x4 = self.conv_4(f4)
        x4 = torch.cat((x4, f4), dim=1)
        x4 = self.adain_4(x4, a4)

        x3 = self.pixel_suffle_3(x4)
        x3 = torch.cat((x3, f3), dim=1)
        x3 = self.adain_3(x3, a3)

        x2 = self.pixel_suffle_2(x3)
        x2 = torch.cat((x2, f2), dim=1)
        x2 = self.adain_2(x2, a2)

        x1 = self.pixel_suffle_1(x2)
        x1 = torch.cat((x1, f1), dim=1)
        x1 = self.adain_1(x1, a1)

        x0 = self.pixel_suffle_0(x1)
        x0 = torch.cat((x0, f0), dim=1)
        x0 = self.adain_0(x0, a0)   
        
        output = self.leaky_relu(x0)
        output = self.conv_last(output)
        output = self.Sigmoid(output)   #10,3,128,128

        if hparams.disc_multiscale == True:
            return (x4, x3, x2, x1, output)
        return output

class Decoder_SPADE(nn.Module):
    def __init__(self, feature_channels, audio_decoder_channels):
        super().__init__()
        self.conv_4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.spade_4 = SPADE(feature_channels[3],feature_channels[3])
        self.adain_4 = AdaIN(feature_channels[3],audio_decoder_channels[3])
        
        # self.pixel_suffle_3 = nn.PixelShuffle(upscale_factor=2)
        self.pixel_suffle_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_3 = torch.nn.Conv2d(feature_channels[3], feature_channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.spade_3 = SPADE(feature_channels[2],feature_channels[2])
        self.adain_3 = AdaIN(feature_channels[2],audio_decoder_channels[2])
        
        # self.pixel_suffle_2 = nn.PixelShuffle(upscale_factor=2)
        self.pixel_suffle_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_2 = torch.nn.Conv2d(feature_channels[2], feature_channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.spade_2 = SPADE(feature_channels[1],feature_channels[1])
        self.adain_2 = AdaIN(feature_channels[1],audio_decoder_channels[1])
        
        # self.pixel_suffle_1 = nn.PixelShuffle(upscale_factor=2)
        self.pixel_suffle_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_1 = torch.nn.Conv2d(feature_channels[1], feature_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.spade_1 = SPADE(feature_channels[0],feature_channels[0])
        self.adain_1 = AdaIN(feature_channels[0],audio_decoder_channels[0])
        
        # self.pixel_suffle_0 = nn.PixelShuffle(upscale_factor=2)
        self.pixel_suffle_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_0 = torch.nn.Conv2d(feature_channels[0], 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.spade_0 = SPADE(8,6)
        self.adain_0 = AdaIN(8,1*80*16)
        
        # Final layer
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.Sigmoid=torch.nn.Sigmoid()
        

    def forward(self, f0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                 a0: Tensor, a1: Tensor, a2: Tensor, a3: Tensor, a4: Tensor):
        x4 = self.conv_4(f4)
        x4 = self.spade_4(x4, f4)
        x4 = self.adain_4(x4, a4)

        x3 = self.pixel_suffle_3(x4)
        x3 = self.conv_3(x3)
        x3 = self.spade_3(x3, f3)
        x3 = self.adain_3(x3, a3)

        x2 = self.pixel_suffle_2(x3)
        x2 = self.conv_2(x2)
        x2 = self.spade_2(x2, f2)
        x2 = self.adain_2(x2, a2)

        x1 = self.pixel_suffle_1(x2)
        x1 = self.conv_1(x1)
        x1 = self.spade_1(x1, f1)
        x1 = self.adain_1(x1, a1)

        x0 = self.pixel_suffle_0(x1)
        x0 = self.conv_0(x0)
        x0 = self.spade_0(x0, f0)
        x0 = self.adain_0(x0, a0)   
        
        output = self.leaky_relu(x0)
        output = self.conv_last(output)
        output = self.Sigmoid(output)   #10,3,128,128

        if hparams.disc_multiscale == True:
            return (x4, x3, x2, x1, output)
        return output

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)
        
    def forward(self, x, r: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        return x, r

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r
    
    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
        
class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):

        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1)
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out

class AdaIN(torch.nn.Module):

    def __init__(self, input_channel, modulation_channel,kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):
        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x




class SPADELayer(torch.nn.Module):
    def __init__(self, input_channel, modulation_channel, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADELayer, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(input_channel)

        self.conv1 = torch.nn.Conv2d(modulation_channel, hidden_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.gamma = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.beta = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input, modulation):
        norm = self.instance_norm(input)
        conv_out = self.conv1(modulation)

        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)

        return norm + norm * gamma + beta


class SPADE(torch.nn.Module):
    def __init__(self, num_channel, num_channel_modulation, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, input, modulations):
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)