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
from my_models import face_encoder, audio_encoder, decoder
from hparams import hparams

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder = face_encoder.FaceEncoder(pretrained=False,in_ch=6)
        self.audio_encoder = audio_encoder.AudioEncoder(pretrained=False,in_ch=1)
        self.decoder = decoder.Decoder([16, 32, 64, 128], [16*40*8, 32*20*4, 64*10*2, 128*5*1])

    #audio: [2, 5, 1, 80, 16]   face: [2, 6, 5, 128, 128]
    def forward(self,audio_sequences: Tensor,face_sequences: Tensor):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])

        f1, f2, f3, f4 = self.face_encoder(face_sequences)#([10, 6, 128, 128])->([10, 16, 64, 64]);([10, 32, 32, 32]);([10, 64, 16, 16]);([10, 128, 8, 8])
        a1, a2, a3, a4 = self.audio_encoder(audio_sequences)#([10, 1, 80, 16])->([10, 16, 40, 8]);([10, 32, 20, 4]);([10, 64, 10, 2]);([10, 128, 5, 1])
        if hparams.disc_multiscale == True:
            outputs_decode = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
            outputs = []
            for i in range(len(outputs_decode)):
                output = outputs_decode[i]
                output = torch.split(output, B, dim=0) 
                output = torch.stack(output, dim=2) 
                outputs.append(output)
            return outputs
        else:    
            output = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
            if input_dim_size > 4:
                output = torch.split(output, B, dim=0) 
                output = torch.stack(output, dim=2) 
            return output

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters 

class Generator_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder = face_encoder.FaceEncoder_CBAM(pretrained=False,in_ch=6)
        self.audio_encoder = audio_encoder.AudioEncoder(pretrained=False,in_ch=1)
        self.decoder = decoder.Decoder([16, 32, 64, 128], [16*40*8, 32*20*4, 64*10*2, 128*5*1])

    #audio: [2, 5, 1, 80, 16]   face: [2, 6, 5, 128, 128]
    def forward(self,audio_sequences: Tensor,face_sequences: Tensor):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])

        f1, f2, f3, f4 = self.face_encoder(face_sequences)#([10, 6, 128, 128])->([10, 16, 64, 64]);([10, 32, 32, 32]);([10, 64, 16, 16]);([10, 128, 8, 8])
        a1, a2, a3, a4 = self.audio_encoder(audio_sequences)#([10, 1, 80, 16])->([10, 16, 40, 8]);([10, 32, 20, 4]);([10, 64, 10, 2]);([10, 128, 5, 1])
        if hparams.disc_multiscale == True:
            outputs_decode = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
            outputs = []
            for i in range(len(outputs_decode)):
                output = outputs_decode[i]
                output = torch.split(output, B, dim=0) 
                output = torch.stack(output, dim=2) 
                outputs.append(output)
            return outputs
        else:    
            output = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
            if input_dim_size > 4:
                output = torch.split(output, B, dim=0) 
                output = torch.stack(output, dim=2) 
            return output

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters 
    
class Generator_SPADE(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder = face_encoder.FaceEncoder(pretrained=False,in_ch=6)
        self.audio_encoder = audio_encoder.AudioEncoder(pretrained=False,in_ch=1)
        self.decoder = decoder.Decoder_SPADE([16, 32, 64, 128], [16*40*8, 32*20*4, 64*10*2, 128*5*1])

    #audio: [2, 5, 1, 80, 16]   face: [2, 6, 5, 128, 128]
    def forward(self,audio_sequences: Tensor,face_sequences: Tensor):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])

        f1, f2, f3, f4 = self.face_encoder(face_sequences)#([10, 6, 128, 128])->([10, 16, 64, 64]);([10, 32, 32, 32]);([10, 64, 16, 16]);([10, 128, 8, 8])
        a1, a2, a3, a4 = self.audio_encoder(audio_sequences)#([10, 1, 80, 16])->([10, 16, 40, 8]);([10, 32, 20, 4]);([10, 64, 10, 2]);([10, 128, 5, 1])
        if hparams.disc_multiscale == True:
            outputs_decode = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
            outputs = []
            for i in range(len(outputs_decode)):
                output = outputs_decode[i]
                output = torch.split(output, B, dim=0) 
                output = torch.stack(output, dim=2) 
                outputs.append(output)
            return outputs
        else:    
            output = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
            if input_dim_size > 4:
                output = torch.split(output, B, dim=0) 
                output = torch.stack(output, dim=2) 
            return output

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters 

class Generator_CBAM_SPADE(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder = face_encoder.FaceEncoder_CBAM(pretrained=False,in_ch=6)
        self.audio_encoder = audio_encoder.AudioEncoder(pretrained=False,in_ch=1)
        self.decoder = decoder.Decoder_SPADE([16, 32, 64, 128], [16*40*8, 32*20*4, 64*10*2, 128*5*1])

    #audio: [2, 5, 1, 80, 16]   face: [2, 6, 5, 128, 128]
    def forward(self,audio_sequences: Tensor,face_sequences: Tensor):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)#([2, 5, 1, 80, 16])->([10, 1, 80, 16])
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)#([2, 6, 5, 512, 512])->([10, 6, 512, 512])

        f1, f2, f3, f4 = self.face_encoder(face_sequences)#([10, 6, 128, 128])->([10, 16, 64, 64]);([10, 32, 32, 32]);([10, 64, 16, 16]);([10, 128, 8, 8])
        a1, a2, a3, a4 = self.audio_encoder(audio_sequences)#([10, 1, 80, 16])->([10, 16, 40, 8]);([10, 32, 20, 4]);([10, 64, 10, 2]);([10, 128, 5, 1])
        output = self.decoder(face_sequences, f1, f2, f3, f4, audio_sequences, a1, a2, a3, a4)#hid([40, 64, 128, 128])
        if input_dim_size > 4:
            output = torch.split(output, B, dim=0) 
            output = torch.stack(output, dim=2) 
        return output

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters 
        
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
