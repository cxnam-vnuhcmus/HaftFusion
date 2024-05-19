import os, random, cv2, argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class HyperCtrolDiscriminator(nn.Module):
    def __init__(self):
        super(HyperCtrolDiscriminator, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            nn.AdaptiveAvgPool2d(1),
            ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        # self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0))
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]#取下半部分

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)#([2, 3, 5, 512, 512])->([10, 3, 512, 512])
        false_face_sequences = self.get_lower_half(false_face_sequences)#([10, 3, 512, 512])->([10, 3, 256, 512])

        false_feats = false_face_sequences#([10, 3, 256, 512])
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)#([10, 32, 256, 512]);([10, 64, 256, 256]);([10, 128, 128, 128]):([10, 256, 64, 64]);([10, 512, 32, 32]);([10, 512, 16, 16]);([10, 512, 14, 14])

        ones = torch.ones((len(false_feats), 1))
        if torch.cuda.is_available():
            ones = ones.cuda()
        binary_feats = self.binary_pred(false_feats).view(len(false_feats), -1)
        false_pred_loss = F.binary_cross_entropy(binary_feats, ones)
        
        return false_pred_loss

    def forward(self, face_sequences, lower_half=True):
        if(len(face_sequences.shape) > 4):
            face_sequences = self.to_2d(face_sequences)#([10, 3, 512, 512])
        if lower_half == True:
            face_sequences = self.get_lower_half(face_sequences)#([10, 3, 512, 512])->([10, 3, 256, 512])

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
    
class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
    
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        input_i = input_nc * 2
        for i in range(num_D):
            input_i = input_i // 2
            if i == num_D - 1:
                input_i = 3
            netD = NLayerDiscriminator(input_i, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)


        cin = 3
        cout = 32
        for i in range(num_D-2,-1,-1):
            conv2d = nonorm_Conv2d(cin, cout, 3, 2, 1)
            setattr(self, 'down' + str(i), conv2d)
            cin = cout
            cout = cout * 2
        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return model(input)

    def forward(self, input, gt=False): #: (B,T,C,H,W)
        # input = torch.cat([input[i,:] for i in range(input.size(0))], dim=0)# : (B*T,C,H,W)
        num_D = self.num_D
        result = []
        if gt == True:
            input_downsampled = input
            for i in range(num_D):
                if self.getIntermFeat:
                    model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                            range(self.n_layers + 2)]
                else:
                    model = getattr(self, 'layer' + str(num_D - 1 - i))
                if i > 0:
                    conv2d = getattr(self, 'down' + str(num_D - 1 - i))
                    input_downsampled = conv2d(input_downsampled)
                result.append(self.singleD_forward(model, input_downsampled))
            result.reverse()
        else:
            for i in range(num_D):
                if self.getIntermFeat:
                    model = [getattr(self, 'scale' + str(i) + '_layer' + str(j)) for j in
                            range(self.n_layers + 2)]
                else:
                    model = getattr(self, 'layer' + str(i))
                result.append(self.singleD_forward(model, input[i]))
        return result
    
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):

        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)