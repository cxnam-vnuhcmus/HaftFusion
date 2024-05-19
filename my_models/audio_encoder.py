from torch import nn
from .mobilenetv3 import MobileNetV3LargeEncoder

class AudioEncoder(nn.Module):
    def __init__(self, pretrained: bool = False,in_ch: int = 3):
        super().__init__()
        self.audio_encoder = MobileNetV3LargeEncoder(pretrained=pretrained,in_ch=in_ch)
        
    def forward(self, input):
        a0, a1, a2, a3 = self.audio_encoder(input)
        return a0, a1, a2, a3